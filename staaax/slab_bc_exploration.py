import warnings
import matplotlib.pyplot as plt
import jax.numpy as jnp
import staaax.angled_stratified as angled_stratified
from diffaaable import aaa
import numpy as onp
from numpy.linalg import LinAlgError
import sax
from staaax.tildify import tildify, inverse_tildify

def mesh_with_branchpoints(ext, ixt, res, branchpoints, num_samples_aaa):
    k_r=jnp.array([])
    for bp in branchpoints:
        k_r = jnp.concat([
            k_r,
            bp-jnp.logspace(ext, ixt, res), 
            bp+jnp.logspace(ixt, ext, res)
        ])

    k_r = jnp.sort(jnp.concat([k_r, -k_r]))

    k_i = jnp.concat([
        -jnp.logspace(ext, ixt, res), 
        jnp.array([0]), 
        jnp.logspace(ixt, ext, res)
    ])

    if SAMPLE_REAL:
        downsample = len(k_r)//num_samples_aaa
    else:
        downsample = int(jnp.sqrt(len(k_r)*len(k_i)//num_samples_aaa))

    K_r, K_i = jnp.meshgrid(k_r, k_i)
    k0_mesh = K_r+ 1j*K_i
    return k0_mesh, downsample

def get_k_r(k0_mesh):
    return k0_mesh[..., :, 0]

def get_borders(k0_mesh):
    k_r = k0_mesh[..., :, 0]
    k_i = k0_mesh[..., 0, :]

    borders = [
        k_r                 + 1j*jnp.min(k_i,axis=-1),
        k_r                 + 1j*jnp.max(k_i,axis=-1),
        jnp.min(k_r,axis=-1)+ 1j*k_i,
        jnp.max(k_r,axis=-1)+ 1j*k_i,
    ]
    return borders

def batch_with_k0(*args):
    broadcastable_args = []

    size = 1
    for i in range(len(args)):
        arg = args[i]
        try:
            arr_arg = jnp.array(arg)
            arr_arg = jnp.atleast_1d(arr_arg)
        except ValueError:
            raise ValueError("Inputs have to be convertible to (jax) numpy arrays")
        if not jnp.ndim(arr_arg) == 1:
            raise ValueError("*args inputs have to be 0D or 1D")
            
        curr_size = arr_arg.shape[0]
        if curr_size != size and size==1:
            size=curr_size
        elif curr_size != size and curr_size != 1:
            raise ValueError(f"detected multiple different sizes in batch dimension {curr_size} vs. {size}")

    one = jnp.ones(size)  
    for arg in args:
        broadcastable_args.append((one*arg)[:, None, None])
    
    return broadcastable_args

def plot_tilde_coord_system(branchpoints, bc_pair, borders, k_r_tilde):
    origin_tilde = tildify(0,  branchpoints, bcs=bc_pair)
    plt.scatter(
        origin_tilde.real,
        origin_tilde.imag,
        marker="o", facecolor="none", s=40,
        linewidth=1, edgecolor="k", zorder=6
    )
    
    for k_border in borders:
        k_tilde_border = tildify(
            k_border, branchpoints, 
            bcs=bc_pair, nan_tolerance=1e-1
        )

        plt.plot(
            k_tilde_border.real, 
            k_tilde_border.imag, 
            color="gray", zorder=6
        )

    plt.plot(
        k_r_tilde.real, 
        k_r_tilde.imag, 
        color="k", zorder=6
    )

def plot_tilde_f(K_tilde, f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tripcolor(
            K_tilde.real.flatten(), 
            K_tilde.imag.flatten(), 
            f.flatten(), 
            norm="log", 
            vmax=1e2, 
            vmin=1e-3,
            rasterized=True,
        )

def approximate_and_plot(z_k, f_k, aaa_tol, branchpoints, 
                         tilde, plot_tilde, color="k"):
    # Make sure no nan in f_k
    filt = ~jnp.isnan(f_k)
    z_k = z_k[filt]
    f_k = f_k[filt]
    
    # Make sure no duplicates in k
    z_k, indices = jnp.unique(z_k, return_index=True)
    f_k = f_k[indices]
    try:

        z_j, f_j, w_j, z_n = aaa(
            z_k, 
            f_k, 
            tol=aaa_tol
        )
        if tilde and not plot_tilde:
            z_n = inverse_tildify(z_n, branchpoints)

        plt.scatter(z_n.real, z_n.imag, zorder=5, marker="x", s=2, color=color)
    except LinAlgError:
        warnings.warn("Failed to find AAA-approximation")

def get_branchpoints(kx, n_sub, n_sup):
    branchpoints = [kx/n_sub, kx/n_sup]
    if not jnp.any(n_sub - n_sup):
        branchpoints=[kx/n_sub]
    return branchpoints

def get_branchcuts(ns):
    if jnp.any(ns[0] - ns[-1]):
        return [
            [ jnp.pi/2,      jnp.pi/2],
            [ jnp.pi/2,   -jnp.pi*3/2],
            [ -jnp.pi*3/2,   jnp.pi/2],
            [ -jnp.pi*3/2,-jnp.pi*3/2],
        ]
        
    return [
        [ jnp.pi/2],
        [ -jnp.pi*3/2],
    ]

def create_batch(kx, ds, ns, ext, ixt, res, num_samples_aaa):
    n_sub, n_sup = ns[0], ns[-1]
    branchpoints = get_branchpoints(kx, n_sub, n_sup)
    k0_mesh, downsample = mesh_with_branchpoints(
        ext, ixt, res, branchpoints, num_samples_aaa
    )

    kx, *dns = batch_with_k0(kx, *ds, *ns)
    ds, ns = dns[:len(ds)], dns[len(ds):]
    k0_mesh = k0_mesh[None, :]   
    return kx, ds, ns, k0_mesh, downsample


def plot_batched(kx, ds, ns, pol, k0_mesh, downsample, 
                 tilde=True, plot_tilde=False, sample_real=False, bc_width=1e-2,
                 aaa_tol=1e-7):
    n_sub, n_sup = ns[0], ns[-1]
    branchpoints = get_branchpoints(kx, n_sub, n_sup)
    bc_pairs = get_branchcuts(ns)

    n_sub, n_sup = [jnp.squeeze(ns[i], axis=(-1, -2)) for i in [0, -1]] # remove k0_batching
    stack, info = angled_stratified.stack_smat_kx(ds, ns, k0_mesh, kx, pol=pol)
    k_r = get_k_r(k0_mesh)
    borders = get_borders(k0_mesh)
    # ------------- Plotting ---------------------

    num = len(ds[0])

    fig, axs = plt.subplots(
        min(3, num), int((num-0.1)//3+1), 
        sharex=True, sharey=True, figsize=(6, 4)
    )
    axs = onp.atleast_2d(onp.array(axs))

    # ------------- Action --------------
    kt_cum = [jnp.array([]) for idx in range(num)]
    tr_cum = [jnp.array([]) for idx in range(num)]

    for i, bc_pair in enumerate(bc_pairs):
        # ------- Set up brances ----------
        settings = sax.get_settings(stack)
        settings = sax.update_settings(
            settings, "if_0", 
            bc_angle_i=bc_pair[0], bc_width_i=bc_width
        )
        settings = sax.update_settings(
            settings, f"if_{len(ds)}", 
            bc_angle_j=bc_pair[-1], bc_width_j=bc_width
        )
        # settings_real = sax.update_settings(
        #     settings, k0=k_r
        # )

        # ------- Evaluate ----------
        smat_mesh = stack(**settings)
        #smat_real = stack(**settings_real)
        
        trans_batch      = smat_mesh[('in', 'out')]
        #trans_real_batch = smat_real[('in', 'out')][:, 0]

        for idx in range(num): 
            # Go through batch dimension (e.g thickness or kx)
            plt.sca(axs.flatten()[idx])
            branchpoints = get_branchpoints(jnp.squeeze(kx[idx]), n_sub[idx], n_sup[idx])

            K_tilde = tildify(k0_mesh[idx], branchpoints, bcs=bc_pair) 
            k_r_tilde = tildify(k_r, branchpoints, bcs=bc_pair)

            trans = trans_batch[idx]
            #trans_real = trans_real_batch[idx]

            if plot_tilde:
                plot_tilde_coord_system(branchpoints, bc_pair, borders, k_r_tilde)
                plot_tilde_f(K_tilde, jnp.abs(trans))

            elif not i: 
                plt.pcolormesh(
                    k0_mesh[idx].real, 
                    k0_mesh[idx].imag, 
                    jnp.abs(trans), 
                    norm="log", vmin=1e-3, vmax=20, rasterized=True
                )
                #plt.plot(k_r[min(idx, len(k_r)-1)], jnp.abs(trans_real), "k")

            if tilde:
                # if sample_real:
                #     kt = k_r_tilde[::downsample]
                #     tr = trans_real[::downsample]
                # else:
                    kt = K_tilde[::downsample, ::downsample].flatten()
                    tr =   trans[::downsample, ::downsample].flatten()
            else:
                # if sample_real:
                #     kt = k_r[::downsample]
                #     tr = trans_real[::downsample]
                # else:
                    kt = k0_mesh[idx, ::downsample, ::downsample].flatten()
                    tr =        trans[::downsample, ::downsample].flatten()

            #approximate_and_plot(kt, tr, aaa_tol, branchpoints, color)

            kt_cum[idx] = jnp.concat([kt_cum[idx], kt])
            tr_cum[idx] = jnp.concat([tr_cum[idx], tr])  

    for idx in range(num): 
        plt.sca(axs.flatten()[idx])
        branchpoints = get_branchpoints(kx[idx], n_sub, n_sup)
        approximate_and_plot(kt_cum[idx], tr_cum[idx], aaa_tol, branchpoints, 
                             tilde=tilde, plot_tilde=plot_tilde)

    # ----------------- Plot Formatting ------------------------
    for ax in axs.flatten():
        k0 = k0_mesh[0]
        # ax.set_xlim([jnp.min(k0.real), jnp.max(k0.real)])
        # ax.set_ylim([jnp.min(k0.imag), jnp.max(k0.imag)])
        #ax.set_aspect(True)
        ax.grid()

    ax_idx = num//3//2

    if plot_tilde:
        axs[ax_idx ,0].set_ylabel(r"$\Im\{\tilde{k}\}$")
        axs[-1,ax_idx].set_xlabel(r"$\Re\{\tilde{k}\}$")
    else:
        axs[ax_idx ,0].set_ylabel(r"$\Im\{k\}$")
        axs[-1,ax_idx].set_xlabel(r"$\Re\{k\}$")

if __name__ == "__main__":
    # ------------- Configuration ---------------------
    TILDE = True
    PLOT_TILDE = TILDE and True
    SAMPLE_REAL = False

    name = "batching_tester2"
    pol = "s"

    kx = 3
    t_layer = jnp.linspace(0.1, 0.3, 8)
    n_sub = 2#2
    n_sup = 1
    n_wg = jnp.linspace(3, 3.1, 8)
    bc_width = 2e-2
    num_samples_aaa = 60

    ext = 0.5
    ixt = -2.6# -1.8
    res = 80

    ext_x = 5 # for plotting
    ext_y = 3
    aaa_tol = 1e-7

    # ext_x = 6.5 # for plotting
    # ext_y = 4

    ns = [n_sub, n_wg, n_sup]
    ds = [t_layer]

    kx, ds, ns, k0_mesh, downsample = create_batch(kx, ds, ns, ext, ixt, res, num_samples_aaa)
    plot_batched(kx, ds, ns, pol, k0_mesh, downsample, 
                 tilde=TILDE, plot_tilde=PLOT_TILDE, sample_real=SAMPLE_REAL, 
                 bc_width=bc_width, aaa_tol=aaa_tol)
    plt.savefig(f"examples/out/{name}.pdf")