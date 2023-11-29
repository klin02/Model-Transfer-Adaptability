# name_list="AlexNet AlexNet_BN VGG_16 VGG_19 Inception_BN ResNet_18 ResNet_50 ResNet_152 MobileNetV2"
name_list="ResNet_152 ResNet_50 ResNet_18 MobileNetV2 Inception_BN VGG_19 VGG_16 AlexNet_BN AlexNet"
for name in $name_list; do 
    scancel -n $name
done