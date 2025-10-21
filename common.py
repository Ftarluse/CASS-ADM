import torch
import torch.nn.functional as F
import math
import torchvision
import copy
import numpy as np
import scipy.ndimage as ndimage


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
       
class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None, embedding=128, group_size=8, hw=1296):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        self.hw = hw
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))

        self.hidden = _hidden
        self.group_size = group_size if group_size > 0 else 0
        
        self.body_out = torch.nn.Sequential(
            torch.nn.Linear(_hidden, embedding),
            torch.nn.BatchNorm1d(embedding),
            torch.nn.LeakyReLU(0.2)
        )

        self.tail = torch.nn.Linear(embedding + self.group_size , 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):

        x = self.body(x)

        if self.group_size > 0:
            y = x.view(-1, self.hw, self.hidden)
 
            y = y - y.mean(dim=1, keepdim=True) 
            y = torch.sqrt(y.pow(2).mean(dim=1) + 1e-8)
            
            y = y.view(-1,  self.group_size, self.hidden// self.group_size)
            y = y.mean(dim=-1).repeat(self.hw, 1)

            x = self.body_out(x)
            x = torch.cat([x, y], dim=-1)
        else:
            x = self.body_out(x)  
            
        x = self.tail(x)
        
        return x

class PositionEncoder(torch.nn.Module):
    def __init__(self, in_planes, size=(36, 36), learnable=True):
        super(PositionEncoder, self).__init__()
        self.size = size
        self.learnable = learnable
        if learnable:
            self.pos_embedding = torch.nn.Parameter(torch.zeros(size[0] * size[1], in_planes))
            torch.nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        else:
            pos_embedding = self.positionalencoding2d(in_planes, size[0], size[1]).view(in_planes, -1)   
            self.register_buffer('pos_embedding',  0.035 * pos_embedding.permute(1, 0))
       
    
    def positionalencoding2d(self, D, H, W):
        P = torch.zeros(D, H, W)
        # Each dimension use half of D
        D = D // 2
        div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
        pos_w = torch.arange(0.0, W).unsqueeze(1)
        pos_h = torch.arange(0.0, H).unsqueeze(1)
        P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
        P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
        P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
        P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
        return P
    
    def forward(self, x):

        repeat_size = x.size(0) // self.pos_embedding.size(0)
        x = x + self.pos_embedding.repeat(repeat_size, 1)
        return x


class Projection(torch.nn.Module):

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes

        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Linear(_in, _out))
            
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):
 
        x = self.layers(x)
        return x




class Extractor(torch.nn.Module):
    def __init__(self,
                 backbone,
                 layers_to_extract_from,
                 in_shape=(3, 224, 224),
                 patch_size=3,
                 stride=1,
                 mid_dim=1024,
                 target_dim=1024,
                 device="cuda"):
        """

        [in_shape]: Get detail（useless for ExtraNet model）  --  print(self.ExtraNet.detail)
        """
        super(Extractor, self).__init__()

        self.backbone = backbone
        _model = torch.nn.Sequential()

        _layers = list(layers_to_extract_from)

        for name, module in backbone.named_children():
            _model.add_module(name, module)
            if name in _layers:
                _layers.remove(name)
            if len(_layers) == 0:
                break
        if len(_layers) != 0:
            raise ValueError(f"not Module {_layers}")

        self.net = NetworkFeatureAggregator(_model, layers_to_extract_from, device=device)

        self.patch_maker = PatchMaker(patchsize=patch_size, stride=stride)
        self.preprocessing = Preprocessing(self.net.feature_dimensions(in_shape), mid_dim)
        self.aggregator = Aggregator(target_dim)

        self.detail = {
            "BackBone": _model,
            "Split_shape": [self.net.fd[0][1]//stride, self.net.fd[0][2]//stride],
            "Forward_shape": self.net.fd}
        self.layers_to_extract_from = layers_to_extract_from
        self.target_dim = target_dim

    def forward(self, x, reshape=False):

        x = self.net(x)
        x = [x[layer] for layer in self.layers_to_extract_from]

        x = [self.patch_maker.patchify(x, return_spatial_info=True) for x in x]
        x = self.patch_maker.concat(x)

        x = self.preprocessing(x)
        x = self.aggregator(x)

        if reshape:
            h, w = self.patch_maker.ref_num_patches
            x = x.reshape(-1, h, w, x.size(-1))
        return x
        
# patchcore extractor
class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self,
                 backbone,
                 layers_to_extract_from,
                 device,
                 train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_idx, extract_block = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )

        self.to(self.device)

    def forward(self, images, eval=True) -> dict:
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        self.fd = [_output[layer].shape[1:] for layer in self.layers_to_extract_from]
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]



class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        return None


class LastLayerToExtractReachedException(Exception):
    pass



class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):

        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize, patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    @ staticmethod
    def unpatch_scores(x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def concat(self, features):

        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        self.ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(self.ref_num_patches[0], self.ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], self.ref_num_patches[0], self.ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        return features

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x



class RescaleSegmentor(object):
    def __init__(self,  target_size, smooth=4):
        self.smoothing = smooth
        self.target_size = target_size
    def __call__(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()
            patch_scores = [ndimage.gaussian_filter(patch_score, sigma=self.smoothing) for patch_score in patch_scores]
        return torch.from_numpy(np.stack(patch_scores, axis=0))


if __name__ == "__main__":

    u = torch.rand((8, 3, 384, 288)).cuda()
    model = Extractor(
        backbone=torchvision.models.resnet50(),
        patch_size=4,
        stride=4,
        layers_to_extract_from=["layer2", "layer3"])
    
    print(model(u, reshape=True).shape)













