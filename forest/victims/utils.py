"""Utilites related to training models."""

from ..utils import write
def print_and_save_stats(epoch, current_lr, train_loss, train_acc, predictions,
                         source_adv_acc, source_adv_loss, source_clean_acc, source_clean_loss, suspicion_rate, false_positive_rate, output):
    
    write(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
        f'Training loss: {train_loss:7.4f}, Training accuracy: {train_acc:7.4%} | ', output)

    if predictions is not None:
        valid_loss, valid_acc, valid_acc_target, valid_acc_source = predictions['all']['loss'], predictions['all']['avg'], predictions['target']['avg'], predictions['source']['avg']
        write('------------- Validation -------------', output)
        write(f'Validation  loss : {valid_loss:7.4f} | Validation accuracy: {valid_acc:7.4%}', output)
        write(f'Target val. acc  : {valid_acc_target:7.4%} | Source val accuracy: {valid_acc_source:7.4%}', output)
        for source_class in source_adv_acc.keys():
            backdoor_acc, clean_acc, backdoor_loss, clean_loss = source_adv_acc[source_class], source_clean_acc[source_class], source_adv_loss[source_class], source_clean_loss[source_class]
            if source_class != 'avg':
                write(f'Source class: {source_class}', output)
            else:
                write(f'Average:', output)
            write('Backdoor loss: {:7.4f} | Backdoor acc: {:7.4%}'.format(backdoor_loss, backdoor_acc), output)
            write('Clean    loss: {:7.4f} | Clean    acc: {:7.4%}'.format(clean_loss, clean_acc), output)
        write('--------------------------------------', output)
    
    if false_positive_rate is not None and suspicion_rate is not None:
        write(f'False positive rate: {false_positive_rate:7.4%} | Suspicion rate: {suspicion_rate:7.4%}', output)
