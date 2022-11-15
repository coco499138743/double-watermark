from torchstat import stat
import torchvision.models as models
import archs

model = archs.NestedUNet(1,3,deep_supervision=False)
stat(model, (3, 96, 96))
