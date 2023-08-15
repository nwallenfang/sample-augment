from torchsummary import summary

from sample_augment.models.classifier import DenseNet201, VisionTransformer, EfficientNetV2, ResNet50


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return trainable_params, frozen_params

# Example Usage:


def main():
    for model in [DenseNet201, VisionTransformer, EfficientNetV2, ResNet50]:
        if model is EfficientNetV2:
            m = model(num_classes=10, size='S')
        else:
            m = model(num_classes=10)

        # if isinstance(m, VisionTransformer):
        #     print(summary(m, input_size=(1, 3, 224, 224), device='cpu'))
        # else:
        #     pint(summary(m, input_size=(3, 256, 256), device='cpu'))

        trainable, frozen = count_params(m)
        print(f'-- {model.__name__} --')
        print(f"Trainable params: {trainable}")
        print(f"Frozen params: {frozen}")


if __name__ == '__main__':
    main()
