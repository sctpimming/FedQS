import models

model_with_bn = models.get_MobileNet(IsGN=False)
model_with_gn = models.get_MobileNet(IsGN=True)

model_with_bn.summary()
model_with_gn.summary()

