import numpy as np
import plotly.graph_objects as go


def make_mesh_trace(mesh, opacity=1.0, color='lightgray'):
    v = mesh.vertices
    f = mesh.faces
    vc = None
    if mesh.visual.kind == 'vertex':
        vc = mesh.visual.vertex_colors[:, :3] / 255.0
        vc = ['rgb({},{},{})'.format(*[int(x * 255) for x in c]) for c in vc]
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                      i=f[:, 0], j=f[:, 1], k=f[:, 2],
                      vertexcolor=vc, opacity=opacity, flatshading=False,
                      name='mesh', showscale=False, lighting=dict(ambient=0.55, diffuse=0.6))


def candidate_traces(cand, rank, arrow_len, color):
    p1, p2, pkm, v, u = cand['p1'], cand['p2'], cand['pkm'], cand['v'], cand['u']
    traces = []
    traces.append(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                                mode='lines+markers',
                                line=dict(color=color, width=7),
                                marker=dict(size=4, color=color),
                                name=f'#{rank} {cand["kind"]} Q={cand["Q"]:.2f}',
                                legendgroup=f'g{rank}'))
    vtip = pkm + v * arrow_len
    traces.append(go.Scatter3d(x=[pkm[0], vtip[0]], y=[pkm[1], vtip[1]], z=[pkm[2], vtip[2]],
                                mode='lines', line=dict(color=color, width=4, dash='dot'),
                                showlegend=False, legendgroup=f'g{rank}'))
    traces.append(go.Cone(x=[vtip[0]], y=[vtip[1]], z=[vtip[2]],
                           u=[v[0] * 0.001], v=[v[1] * 0.001], w=[v[2] * 0.001],
                           anchor='tip', sizemode='absolute', sizeref=arrow_len * 0.35,
                           showscale=False, colorscale=[[0, color], [1, color]],
                           showlegend=False, legendgroup=f'g{rank}'))
    return traces


PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
           '#f032e6', '#bcf60c', '#fabebe', '#008080', '#9a6324', '#000075']


def visualize_candidates(mesh, candidates, title, top_n=8, out_html='candidates.html'):
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    arrow_len = diag * 0.18
    fig = go.Figure()
    fig.add_trace(make_mesh_trace(mesh, opacity=0.55))
    shown = candidates[:top_n]
    for i, c in enumerate(shown):
        color = PALETTE[i % len(PALETTE)]
        for t in candidate_traces(c, i + 1, arrow_len, color):
            fig.add_trace(t)
    fig.update_layout(
        title=title, height=820, width=1100,
        scene=dict(aspectmode='data',
                   xaxis_title='x (mm)', yaxis_title='y (mm)', zaxis_title='z (mm)'),
        legend=dict(itemsizing='constant'))
    fig.write_html(out_html)
    return out_html
