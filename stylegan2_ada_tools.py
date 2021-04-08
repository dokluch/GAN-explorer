def make_img_from_seed(Gs, seed_in = 0):
    torch.manual_seed(seed_in)
    z1 = torch.randn([1, Gs.z_dim]).cuda() 
    c = None #class
    w = Gs.mapping(z1, c, truncation_psi=0.7, truncation_cutoff=8)
    img = Gs.synthesis(w, noise_mode='const', force_fp32=True)

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

    return(img)

def make_img_from_vec(Gs, vec_in = 0):
    c = None
    w = Gs.mapping(vec_in, c, truncation_psi=0.7, truncation_cutoff=8)
    img = Gs.synthesis(w, noise_mode='const', force_fp32=True)

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

    return(img)
    
def seed2vec(Gs, seed_in = 0):
    torch.manual_seed(seed_in)
    z = torch.randn([1, Gs.z_dim]).cuda()
    return z

def generate_image(Gs, z, truncation_psi):
    c = None
    w = Gs.mapping(z, c, truncation_psi=0.7, truncation_cutoff=8)
    img = Gs.synthesis(w, noise_mode='const', force_fp32=True)

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    return img