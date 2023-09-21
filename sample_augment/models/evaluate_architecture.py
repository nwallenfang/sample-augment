from typing import List

from sample_augment.models.evaluate_baseline import check_architecture_reports
from sample_augment.models.evaluate_classifier import KFoldClassificationReport, ClassificationReport


def evaluate_architecture():
    reports: List[List[ClassificationReport]] = [
        KFoldClassificationReport.from_name('DenseNet-naive_36fbb6').reports,
        KFoldClassificationReport.from_name('ResNet-naive_6d3d80').reports,
        KFoldClassificationReport.from_name('EfficientNetS-naive_373365').reports,
        KFoldClassificationReport.from_name('EfficientNetL-naive_21b34d').reports,
        KFoldClassificationReport.from_name('ViT-naive_758666.json').reports,
    ]
    names = ['DenseNet', 'ResNet', 'EfficientNet-S', 'EfficientNet-L', 'VisionTransformer']
    check_architecture_reports(names, reports)


if __name__ == '__main__':
    evaluate_architecture()
