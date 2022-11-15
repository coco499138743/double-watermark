from torchstat import stat
import torchvision.models as models
import archs

model = archs.UNet(1,3)
stat(model, (3, 96, 96))
