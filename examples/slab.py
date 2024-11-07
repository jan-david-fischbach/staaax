import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import angled_stratified
from diffaaable import aaa
from examples import material
from propagation import angled_sqrt
from scipy.constants import c as c0
import numpy as onp
from numpy.linalg import LinAlgError
import sax

def tildify(k, Cs, bcs, nan_tolerance=0):
    return 1/len(Cs)*jnp.sum(jnp.array([
        angled_sqrt(
            (jnp.squeeze(k)**2 - jnp.squeeze(C)**2), 
            bc_angle=bc, 
            nan_tolerance=nan_tolerance) for C, bc in zip(Cs, bcs)
    ]), axis=0)

# def tildify_kx(k, ns, kx, bcs):
#     return 1/len(ns)*jnp.sum(jnp.array([
#         angled_sqrt(
#             ((jnp.squeeze(k)*n)**2 - jnp.squeeze(kx)**2), 
#             bc_angle=bc, 
#             nan_tolerance=0) for n, bc in zip(ns, bcs)
#     ]), axis=0)

if __name__ == "__main__":

    pol = "s"
    pol_idx = 1 if pol=="p" else 0
    kx = 3
    n_sub = 2
    n_sup = 1
    n_wg = 4
    bc_width = 2e-2

    TILDE = True
    SAMPLE_REAL = False
    num_samples = 300

    ext = 0.7
    ixt = -3
    res = 120
    k_r = jnp.concat([
        kx/n_sup-jnp.logspace(ext, ixt, res), kx/n_sup+jnp.logspace(ixt, ext, res),
        kx/n_sub-jnp.logspace(ext, ixt, res), kx/n_sub+jnp.logspace(ixt, ext, res),
    ])

    k_r = jnp.sort(jnp.concat([k_r, -k_r]))

    k_i = jnp.concat([
        -jnp.logspace(ext, ixt, res), jnp.array([0]), jnp.logspace(ixt, ext, res)
    ])

    borders = [
        k_r     + 1j*min(k_i),
        k_r     + 1j*max(k_i),
        min(k_r)+ 1j*k_i,
        max(k_r)+ 1j*k_i,
    ]

    if SAMPLE_REAL:
        downsample = len(k_r)//num_samples
    else:
        downsample = int(jnp.sqrt(len(k_r)*len(k_i)//num_samples))


    K_r, K_i = jnp.meshgrid(k_r, k_i)

    t_layer = jnp.linspace(0.02, 0.3, 9)
    #t_layer = jnp.linspace(0.2, 0.2, 1)
    #t_layer = jnp.linspace(0.264, 0.272, 12)

    kx = jnp.ones_like(t_layer)*kx
    k0_mesh = K_r+ 1j*K_i

    k0_mesh = k0_mesh[None, :]
    kx = kx[:, None, None]
    t_layer = t_layer[:, None, None]

    ns = [n_sub, n_wg+1e-4j, n_sup]
    ds = [t_layer]
    tildify_ns = [n_sub, n_sup]

    num = len(t_layer)
    fig, axs = plt.subplots(min(3, num), int((num-0.1)//3+1), sharex=True, sharey=True)
    if num==1:
        axs = onp.array([axs])

    bc = 0*jnp.pi
    stack, info = angled_stratified.stack_smat_kx(ds, ns, k0_mesh, kx, pol=pol)
    
    bcs = [-jnp.pi*3/2, -jnp.pi*5/4, -jnp.pi/2, jnp.pi/2, jnp.pi*3/2]

    bc_pairs = [
        [ -jnp.pi*3/2,-jnp.pi*3/2],
        [ jnp.pi/2,   -jnp.pi*3/2],
        [ -jnp.pi*3/2,   jnp.pi/2],
        [ jnp.pi/2,      jnp.pi/2]
    ]

    kt_cum = jnp.array([])
    tr_cum = jnp.array([])
    for i, bc_pair in enumerate(bc_pairs):
        #plt.figure()
        settings = sax.get_settings(stack)

        settings = sax.update_settings(
            settings, "if_0", 
            bc_angle_i=bc_pair[0], bc_width_i=bc_width
        )

        settings = sax.update_settings(
            settings, f"if_{len(ds)}", 
            bc_angle_j=bc_pair[1], bc_width_j=bc_width
        )

        smat_mesh = stack(**settings)

        settings = sax.update_settings(
            settings, k0=k_r
        )
        smat_real = stack(**settings)

        for idx, ax in enumerate(axs.flatten()):
            branchpoints = [kx[idx]/n_sub, kx[idx]/n_sup]
            if n_sub == n_sup:
                branchpoints=[kx[idx]/n_sub]
            plt.sca(ax)
            
            trans      = smat_mesh[('in', 'out')][idx]
            trans_real = smat_real[('in', 'out')][idx][0]

            # K_tilde =   tildify_kx(k0_mesh, tildify_ns, kx[idx], bcs=bc_pair) 
            # k_r_tilde = tildify_kx(k_r, tildify_ns, kx[idx], bcs=bc_pair)

            K_tilde = tildify(k0_mesh, branchpoints, bcs=bc_pair) 
            k_r_tilde = tildify(k_r, branchpoints, bcs=bc_pair)
            origin_tilde = tildify(0, branchpoints, bcs=bc_pair)
            
            if TILDE:
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

                plt.scatter(
                    origin_tilde.real,
                    origin_tilde.imag,
                    marker="o", facecolor="none", s=40,
                    linewidth=1, edgecolor="k", zorder=6
                )
                if SAMPLE_REAL:
                    kt = k_r_tilde[::downsample]
                    tr = trans_real[::downsample]
                else:
                    kt = K_tilde[::downsample, ::downsample].flatten()
                    tr =   trans[::downsample, ::downsample].flatten()
            else:
                if SAMPLE_REAL:
                    kt = k_r[::downsample]
                    tr = trans_real[::downsample]
                else:
                    kt = k0_mesh[0, ::downsample, ::downsample].flatten()
                    tr =   trans[::downsample, ::downsample].flatten()
            
            filt = ~jnp.isnan(tr)
            kt = kt[filt]
            tr = tr[filt]

            kt_cum = jnp.concat([kt_cum, kt])
            tr_cum = jnp.concat([tr_cum, tr])

            try:
                z_j, f_j, w_j, z_n = aaa(
                    kt, 
                    tr, 
                    tol=1e-7
                )
            except LinAlgError:
                z_n = jnp.zeros(0)

            if TILDE:
                # plt.pcolormesh(
                #         K_tilde.real, K_tilde.imag, 
                #         jnp.abs(trans), 
                #         norm="log", /home/jd/bin/mambaforge/envs/ag-surmof/bin/python /home/jd/phd/code/propagate_fp/examples/slab.py
                #         vmax=1e2, 
                #         vmin=1e-3
                # )

                plt.tripcolor(
                        K_tilde.real.flatten(), 
                        K_tilde.imag.flatten(), 
                        jnp.abs(trans).flatten(), 
                        norm="log", 
                        vmax=1e2, 
                        vmin=1e-3)
                
                # plt.scatter(
                #         K_tilde.real.flatten(), K_tilde.imag.flatten(), 
                #         c = jnp.abs(trans), 
                #         norm="log", 
                #         vmax=1e2, 
                #         vmin=1e-3
                # )
            else: 
                plt.pcolormesh(K_r, K_i, jnp.abs(trans), norm="log", vmin=1e-3, vmax=3)
            
            if not TILDE and bc_pair[1] == bc_pair[0] == jnp.pi/2:
                plt.plot(k_r, jnp.abs(trans_real), "k")
            plt.grid()
            
            # color=f"C{i}"
            # plt.scatter(z_n.real, z_n.imag, zorder=5, marker="x", color=color)
            # plt.scatter(kt.real, kt.imag, zorder=5, marker=".", color=color)
        #plt.show(block=False)
    try:
        z_j, f_j, w_j, z_n = aaa(
            kt_cum, 
            tr_cum, 
            tol=1e-7
        )
        plt.scatter(z_n.real, z_n.imag, zorder=5, marker="x", color="k")
    except LinAlgError:
        pass

    for ax in axs.flatten():
        ext_x = 5
        ext_y = 4
        ax.set_xlim([-ext_x, ext_x])
        ax.set_ylim([-ext_y, ext_y])
        ax.set_aspect(True)


    ax_idx = num//3//2
    if axs.ndim == 1:
        axs = axs[:, None]
    if TILDE:
        axs[ax_idx ,0].set_ylabel(r"$\Im\{\tilde{k}\}$")
        axs[-1,ax_idx].set_xlabel(r"$\Re\{\tilde{k}\}$")
    else:
        axs[ax_idx ,0].set_ylabel(r"$\Im\{k\}$")
        axs[-1,ax_idx].set_xlabel(r"$\Re\{k\}$")
        plt.xticks([-3, 0, 3])
    plt.show()