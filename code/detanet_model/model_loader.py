import torch
from .detanet import DetaNet
def scalar_model(device,params='trained_param/qm7x/energy.pth',max_number=9):
    state_dict = torch.load(params)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=max_number,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=True,
                    norm=False,
                    out_type='scalar',
                    grad_type=None,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def force_model(device,params='trained_param/qm7x/force.pth'):
    state_dict = torch.load(params)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=17,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=True,
                    norm=False,
                    out_type='scalar',
                    grad_type='force',
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def charge_model(device,params='trained_param/qm9spectra/npacharge.pth'):
    state_dict=torch.load(params)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=1,
                 irreps_out=None,
                 summation=False,
                 norm=False,
                 out_type='scalar',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def dipole_model(device,params='trained_param/qm9spectra/dipole.pth'):
    state_dict=torch.load(params)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=1,
                 irreps_out='1o',
                 summation=True,
                 norm=False,
                 out_type='dipole',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def polar_model(device,params='trained_param/qm9spectra/polar.pth'):
    state_dict=torch.load(params)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='2e',
                 summation=True,
                 norm=False,
                 out_type='2_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def quadrupole_model(device,params='trained_param/qm9spectra/quadrupole.pth'):
    state_dict=torch.load(params)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='2e',
                 summation=True,
                 norm=False,
                 out_type='2_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def hyperpolar_model(device,params='trained_param/qm9spectra/hyperpolar.pth'):
    state_dict=torch.load(params)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='1o+3o',
                 summation=True,
                 norm=False,
                 out_type='3_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def octapole_model(device,params='trained_param/qm9spectra/octapole.pth'):
    state_dict=torch.load(params)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='1o+3o',
                 summation=True,
                 norm=False,
                 out_type='3_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def Hi_model(device,params='trained_param/qm9spectra/Hi.pth'):
    state_dict = torch.load(params)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=9,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='scalar',
                    grad_type='Hi',
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def Hij_model(device,params='trained_param/qm9spectra/Hij.pth'):
    state_dict = torch.load(params)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=9,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='scalar',
                    grad_type='Hij',
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def dedipole_model(device,params='trained_param/qm9spectra/dedipole.pth'):
    state_dict=torch.load(params)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=1,
                 irreps_out='1o',
                 summation=False,
                 norm=False,
                 out_type='dipole',
                 grad_type='dipole',
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def depolar_model(device,params='trained_param/qm9spectra/depolar.pth'):
    state_dict=torch.load(params)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='2e',
                 summation=False,
                 norm=False,
                 out_type='2_tensor',
                 grad_type='polar',
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def nmr_model(device,params):
    state_dict = torch.load(params)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=9,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='scalar',
                    grad_type=None,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def uv_model(device,params='trained_param/qm9spectra/borden_os.pth'):
    state_dict = torch.load(params)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=9,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=240,
                    irreps_out=None,
                    summation=True,
                    norm=False,
                    out_type='scalar',
                    grad_type=None,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model


def approach_1_QM9SPol_model(device,params='trained_param/dynamic-polarizability/OPT_QM9SPol_NoSpectra0features70epochs_64batchsize_0.0005lr_6cutoff_6numblock_128features_KITQM9.pth'):
    state_dict = torch.load(params)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=6,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=32,
                    rc=6.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=34,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=(4*62),
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='124x2e',
                    grad_type=None,
                    x_features=0,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def approach_2_QM9SPol_model(device,params='trained_param/dynamic-polarizability/OPT_QM9SPol_Spectra62features70epochs_32batchsize_0.0005lr_6cutoff_6numblock_256features_KITQM9.pth'):
    state_dict = torch.load(params)
    model = DetaNet(num_features=256,
                    act='swish',
                    maxl=3,
                    num_block=6,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=32,
                    rc=6.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=34,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=(4*62),
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='124x2e',
                    grad_type=None,
                    x_features=62,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model



def approach_1_HOPV_model(device,params='trained_param/dynamic-polarizability/OPT_HOPV_NoSpectra0spectra80epochs_32batchsize_0.0005lr_4cutoff_4numblock_64features_8att_HOPV.pth'):
    state_dict = torch.load(params)
    model = DetaNet(num_features=64,
                    act='swish',
                    maxl=3,
                    num_block=4,
                    radial_type='trainable_bessel',
                    num_radial=128,
                    attention_head=8,
                    rc=4.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=34,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=(4*62),
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='124x2e',
                    grad_type=None,
                    x_features=0,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model


def approach_2_HOPV_model(device,params='trained_param/dynamic-polarizability/OPT_HOPV_Spectra62spectra80epochs_8batchsize_0.001lr_4cutoff_4numblock_256features_64att_HOPV.pth'):
    state_dict = torch.load(params)
    model = DetaNet(num_features=256,
                    act='swish',
                    maxl=3,
                    num_block=4,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=64,
                    rc=4.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=34,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=(4*62),
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='124x2e',
                    grad_type=None,
                    x_features=62,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model
