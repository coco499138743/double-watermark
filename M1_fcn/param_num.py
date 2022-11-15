from torchstat import stat
import torchvision.models as models
import archs

vgg_model = archs.VGGNet(requires_grad=True, show_params=False)
model = archs.FCNs(pretrained_net=vgg_model, n_class=1)
stat(model, (3, 96, 96))
