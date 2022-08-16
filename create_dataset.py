from dataset.uci_har import UCI_HAR
from dataset.usc_had import USC_HAD
from dataset.pamap2 import PAMAP2
from dataset.opportunity import OPPORTUNITY
from dataset.mhealth import MHEALTH
from dataset.mmshd import MMHAD
from dataset.mobiact import MobiAct


if __name__ == "__main__":
    dataset = UCI_HAR()
    dataset.dataset_verbose()
    dataset.save_split()

    dataset = MobiAct(clean=True, fall=False)
    dataset.dataset_verbose()
    dataset.save_split('splits_Xfall')
    
    dataset = PAMAP2(clean=False, include_null=True)
    dataset.save_split('splits_Xclean')

    dataset = MHEALTH(clean=False, include_null=True)
    dataset.dataset_verbose()
    dataset.save_split('splits_Xclean')

#    dataset = USC_HAD()
#    dataset.dataset_verbose()
#    dataset.save_split()
#
#    clean = [True, False]
#    include_null = [True, False]
#    for c in clean:
#        for n in include_null:
#            name = 'splits'
#            if not c: 
#                name = name + '_Xclean'
#            if not n:
#                name = name + '_Xnull' 
#            dataset = OPPORTUNITY(clean=c, include_null=n)
#            dataset.save_split(name)
#            dataset = PAMAP2(clean=c, include_null=n)
#            dataset.save_split(name)
#            dataset = MHEALTH(clean=c, include_null=n)
#            dataset.save_split(name)