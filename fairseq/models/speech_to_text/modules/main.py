import torch
from torch.utils.data import DataLoader
from fairseq.models.speech_to_text.modules.config import Params
#from fairseq.models.speech_to_text.modules.src.decoder import CerWer
import wandb

from examples.simultaneous_translation.models.convtransformer_simul_trans import (
    AugmentedMemoryConvTransformerModel
)
from examples.speech_recognition.infer import main
from fairseq.tasks.simultaneous_translation import (
    SimulSpeechToTextTask
)

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

args = Params()
task = SimulSpeechToTextTask.setup_task(args)

task.load_dataset(split="train-clean-100")
task.load_dataset(split="test-clean")
train_dataset = task.datasets["train-clean-100"]
test_dataset = task.datasets["test-clean"]

train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              collate_fn=train_dataset.collater)
test_dataloader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=test_dataset.collater)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
model = AugmentedMemoryConvTransformerModel.build_model(args, task)

if args.from_pretrained:
    model.load_state_dict(torch.load(args.model_path))

model.to(device)
smooth_crit = LabelSmoothedCrossEntropyCriterion.build_criterion(args, task)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
num_steps = len(train_dataloader) * args.num_epochs
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00001)
#cerwer = CerWer()

if args.wandb_log:
    wandb.init(project=args.wandb_name)
#    wandb.watch(model, log="all", log_freq=1000)

def to_gpu(sample, device):
    new_sample = dict()
    for k, v in sample.items():
        if isinstance(v, int):
            new_sample[k] = v
        elif k == "net_input":
            new_sample[k] = dict()
            for k1, v1 in sample[k].items():
                new_sample[k][k1] = v1.to(device)
        else:
            new_sample[k] = v.to(device)
    return new_sample

start_epoch = args.start_epoch + 1 if args.from_pretrained else 1
for epoch in range(start_epoch, args.num_epochs + 1):
#    train_cer, train_wer, val_wer, val_cer = 0.0, 0.0, 0.0, 0.0
    train_loss = 0.0
    model.train()
    for i, sample in enumerate(train_dataloader):
        optimizer.zero_grad()
        sample = to_gpu(sample, device)
        smooth_loss, smooth_sample_size, smooth_logging_output = smooth_crit(
            model, sample
        )
        smooth_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        train_loss += smooth_loss.item()
        # _, max_probs = torch.max(outputs, 2)
        # train_epoch_cer, train_epoch_wer, train_decoded_words, train_target_words = cerwer(max_probs.T.cpu().numpy(),
        #                                                                                    targets.cpu().numpy(),
        #                                                                                    inputs_length,
        #                                                                                    targets_length)
        # train_wer += train_epoch_wer
        # train_cer += train_epoch_cer
        if (i + 1) % 100 == 0:
            wandb.log({"train_loss": train_loss / (args.batch_size * 100)})
            train_loss = 0.0

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, sample in enumerate(test_dataloader):
            sample = to_gpu(sample, device)
            smooth_loss, smooth_sample_size, smooth_logging_output = smooth_crit(
                model, sample
            )
            # loss = criterion(outputs.log_softmax(dim=2), targets, inputs_length, targets_length).cpu()
            val_loss += smooth_loss.item()
            # _, max_probs = torch.max(outputs, 2)
            # val_epoch_cer, val_epoch_wer, val_decoded_words, val_target_words = cerwer(max_probs.T.cpu().numpy(),
            #                                                                            targets.cpu().numpy(),
            #                                                                            inputs_length, targets_length)
            # val_wer += val_epoch_wer
            # val_cer += val_epoch_cer
        wandb.log({"val_loss": val_loss / (args.batch_size * len(test_dataloader))})
    torch.save(model.state_dict(), f"""left{args.left_context}_right{
                    args.right_context}_segment{args.segment_size}_epoch{epoch}.pth""")
    print("model saved")
    # if params["wandb_log"]:
    #     wandb.log({"train_loss": np.mean(train_losses),
    #                # "val_wer": val_wer / len(test_dataset),
    #                # "train_cer": train_cer / len(train_dataset),
    #                "val_loss": np.mean(val_losses),
    #                # "train_wer": train_wer / len(train_dataset),
    #                # "val_cer": val_cer / len(test_dataset),
    #                # "train_samples": wandb.Table(columns=["Target text", "Predicted text"],
    #                #                              data=[train_target_words, train_decoded_words]),
    #                # "val_samples": wandb.Table(columns=["Target text", "Predicted text"],
    #                #                            data=[val_target_words, val_decoded_words]),
    #                })


