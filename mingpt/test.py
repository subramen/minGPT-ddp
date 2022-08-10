import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig):
    print(cfg)

if __name__ == "__main__":
    main()