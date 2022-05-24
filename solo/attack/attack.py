import torch
import numpy as np
import torch.nn as nn
from solo.attack import* 

class AttackGenerate(object):
    _ATTACKS = {
        'fgsm': FGSM,
        # 'bim': BIM,
        'pgd': PGD,
        # 'mim': MIM,
        'cw': CW,
        # 'deepfool': DeepFool,
        # 'dim': DI2FGSM,
        'nes': NES,
    } 
    def __init__(
        self,
        attack_method, 
        attack_net,
        distance, 
        loss, 
        target: bool, 
        data_name,
        **kwargs,
    ): 

        #print(kwargs)
        """ kwargs Args:
        eps,                #bim, pgd, mim, dim
        stepsize,         #bim, pgd, mim, dim
        steps,              #bim, pgd, mim, dim
        decay_factor,       #mim, dim
        resize_rate,        #dim
        diversity_prob,     #dim
        overshoot,          #deepfool
        max_steps,          #deepfool
        max_iter,           #cw
        binary_search_steps,    #cw
        lr,                 #cw
        init_const,         #cw
        kappa,              #cw (confidence)
        """

        super().__init__()

        self.attack_method = attack_method
        self.attack_net = attack_net
        self.distance = distance
        self.loss = loss
        self.target = target
        self.data_name = data_name

        if distance == 'L2':
            dist = 2
        elif distance ==' Linf':
            dist = np.inf
        else:
            dist = 2

        #initial attack
        self.attack = self.generate_attacker(attack_method, attack_net, dist, data_name, loss, target, kwargs)
        print("finish prepare transform")

    def _attack(self, batch):
        """Basic forward that allows children classes to override forward().

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y]

        Returns:
            Dict: dict of logits and features.
        """
        #print(batch)

        X, labels, targets = batch

        adv_img = []
        # print("X", len(X))

        for x in X:
            # print(len(x))
            adv_img += [self.attack.forward(x, labels, targets)]

        # print("adv_img", len(adv_img))
        
        return adv_img
    
    def generate_attacker(self, attack_name, net, distance, dataset, loss, target, kwargs):
        if attack_name == 'fgsm':
            attack_class = self._ATTACKS[attack_name]
            attack = attack_class(net, p=distance, eps=kwargs['eps'], data_name=dataset,target=target, loss=loss)
        elif attack_name == 'bim':
            attack_class = self._ATTACKS[attack_name]
            attack = attack_class(net, epsilon=kwargs['eps'], p=distance, stepsize=kwargs['stepsize'], steps=kwargs['steps'], data_name=dataset,target=target, loss=loss)
        elif attack_name == 'pgd':
            attack_class = self._ATTACKS[attack_name]
            attack = attack_class(net, epsilon=kwargs['eps'], norm=distance, stepsize=kwargs['stepsize'], steps=kwargs['steps'], data_name=dataset,target=target, loss=loss)
        elif attack_name == 'mim':
            attack_class = self._ATTACKS[attack_name]
            attack = attack_class(net, epsilon=kwargs['eps,'], p=distance, stepsize=kwargs['stepsize'], steps=kwargs['steps'], decay_factor=kwargs['decay_factor'], 
                                    data_name=dataset, target=target, loss=loss)
        elif attack_name == 'cw':
            attack_class = self._ATTACKS[attack_name]
            attack = attack_class(net ,distance, target, kwargs['kappa'], kwargs['cw_lr'], kwargs['init_const'], kwargs['max_iter'], 
                                    kwargs['binary_search_steps'], dataset)
        elif attack_name == 'deepfool':
            attack_class = self._ATTACKS[attack_name]
            attack = attack_class(net, kwargs['overshoot'], kwargs['max_steps'], distance, target)
        elif attack_name == 'dim':
            attack_class = self._ATTACKS[attack_name]
            attack = attack_class(net, p=distance, eps=kwargs['eps'], stepsize=kwargs['stepsize'], steps=kwargs['steps'], decay=kwargs['decay_factor'], 
                                    resize_rate=kwargs['resize_rate'], diversity_prob=kwargs['diversity_prob'], data_name=dataset,target=target, loss=loss)
        elif attack_name == 'nes':
            attack_class = self._ATTACKS[attack_name]
            attack = attack_class(net, nes_samples=kwargs['nes_samples'], sample_per_draw=kwargs['nes_per_draw'], 
                                    p=distance, max_queries=kwargs['bt_max_queries'], epsilon=kwargs['eps'], step_size=kwargs['stepsize'],
                                    data_name=dataset, search_sigma=0.02, decay=1.0, random_perturb_start=True, target=target)
        return attack