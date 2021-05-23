from TrainingHelper import TrainingHelper

if __name__ == "__main__":
    # starting training from if __name__ == "__main__" block 
    # as the dataloader has some bugs which throws error when not initiated within this block.
    # Ref: https://github.com/pytorch/pytorch/issues/2341#issuecomment-346551098

    trainer = TrainingHelper()

    trainer.train()
    # restore_ckpt = "./summaries/2021_05_22_17_12_01/ckpt/model_eps_2_test_loss_1.4546.pth"
    # trainer.train(resume=True, resume_ckpt=restore_ckpt)
    # resnet_ckpt = "./pretrained_ckpt/resnet18-5c106cde.pth"
    # trainer.train(pretrained_ckpt=resnet_ckpt)

