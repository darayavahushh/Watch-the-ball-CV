from datasets_.cocoMini import get_dataloaders

CFG='configs/config.yaml'

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(CFG)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Grab one batch
    images, targets = next(iter(train_loader))
    print("Batch image tensor shape:", images[0].shape)
    print("Batch target example:", targets[0])

