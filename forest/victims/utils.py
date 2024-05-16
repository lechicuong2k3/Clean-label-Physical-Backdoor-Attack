"""Utilites related to training models."""

from ..utils import write
def print_and_save_stats(epoch, current_lr, train_loss, train_acc, predictions,
                         source_adv_acc, source_adv_loss, source_clean_acc, source_clean_loss, suspicion_rate, false_positive_rate, output):
    
    write(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
        f'Training loss: {train_loss:7.4f}, Training accuracy: {train_acc:7.4%} | ', output)

    if predictions is not None:
        valid_loss, valid_acc, valid_acc_target, valid_acc_source = predictions['all']['loss'], predictions['all']['avg'], predictions['target']['avg'], predictions['source']['avg']
        write('------------- Validation -------------', output)
        write(f'Validation  loss: {valid_loss:7.4f} | Validation accuracy: {valid_acc:7.4%}', output)
        write(f'Target val. acc : {valid_acc_target:7.4%} | Source val accuracy: {valid_acc_source:7.4%}', output)
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
        
def analyze_and_print(stats, output):
    write('\n---------------- Summary -------------------', output)
    import statistics
    
    # Validation Accuracy
    valid_acc = stats['valid_acc']
    if len(valid_acc) == 1:
        valid_acc = [valid_acc[0], valid_acc[0], valid_acc[0]]
    acc_mean = statistics.mean(valid_acc)
    acc_std = statistics.stdev(valid_acc)
    write('Valid Accuracy: {:.1%} ({:.1%})'.format(acc_mean, acc_std), output)
    
    # Backdoor and Clean Accuracy
    for source_class in stats['backdoor_acc'].keys():
        backdoor_acc, clean_acc = stats['backdoor_acc'][source_class], stats['clean_acc'][source_class]
        if source_class != 'avg':
            write(f'Source class: {source_class}', output)
        else:
            if len(stats['backdoor_acc'].keys()) == 2: continue
            write(f'Average:', output)
            
        if len(backdoor_acc) == 1:
            backdoor_acc = [backdoor_acc[0], backdoor_acc[0], backdoor_acc[0]]
        if len(clean_acc) == 1:
            clean_acc = [clean_acc[0], clean_acc[0], clean_acc[0]]
            
        backdoor_acc_mean = statistics.mean(backdoor_acc)
        backdoor_acc_std = statistics.stdev(backdoor_acc)
        write('Backdoor acc: {:.1%} ({:.1%})'.format(backdoor_acc_mean, backdoor_acc_std), output)
        
        clean_acc_mean = statistics.mean(clean_acc)
        clean_acc_std = statistics.stdev(clean_acc)
        write('Clean    acc: {:.1%} ({:.1%})'.format(clean_acc_mean, clean_acc_std), output)
    
    # Suspicion Rate
    suspicion_rate = stats['sr']
    if len(suspicion_rate) == 1:
        suspicion_rate = [suspicion_rate[0], suspicion_rate[0], suspicion_rate[0]]
    sr_mean = statistics.mean(suspicion_rate)
    sr_std = statistics.stdev(suspicion_rate)
    write('Suspicion rate: {:.1%} ({:.1%})'.format(sr_mean, sr_std), output)
    
    # False positive rate
    false_positive_rate = stats['fpr']
    if len(false_positive_rate) == 1:
        false_positive_rate = [false_positive_rate[0], false_positive_rate[0], false_positive_rate[0]]
    fpr_mean = statistics.mean(false_positive_rate)
    fpr_std = statistics.stdev(false_positive_rate)
    write('False positive rate: {:.1%} ({:.1%})'.format(fpr_mean, fpr_std), output)
    write('--------------------------------------------', output)