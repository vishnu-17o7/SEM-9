# Place training/evaluation datasets here.

Each experiment looks for datasets in this folder and prints an exact
`FileNotFoundError` hint if a required file is missing.

Suggested layouts:

- `fashion_mnist/`     — Program 11 (auto-cached from keras/torchvision)
- `cifar10/`           — Program 12 (auto-cached from torchvision)
- `eurosat/`           — Program 12 (auto-cached from torchvision)
- `lfw_subset/<name>/` — Program 10 (auto-fetched by sklearn as fallback)
- `mvtec_subset/<class>/<split>/<image>` — Program 7
