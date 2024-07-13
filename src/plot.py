import numpy as np
import pandas as pd
import matplotlib as mpl

from mce.util.plot_base import PlotBase

class MyPlot(PlotBase):
    def __init__(self, **kw):
        super().__init__(**kw)

        self.map_probability = {
            0.05: 'very_likely__lower',
            0.17: 'likely__lower',
            0.5: 'central',
            0.83: 'likely__upper',
            0.95: 'very_likely__upper',
        }
        self.map_probability_q = {
            'Q05': 'very_likely__lower',
            'Q17': 'likely__lower',
            'Q50': 'central',
            'Q83': 'likely__upper',
            'Q95': 'very_likely__upper',
        }
        self.map_scenario = {
            'ssp119': 'SSP1-1.9',
            'ssp126': 'SSP1-2.6',
            'ssp245': 'SSP2-4.5',
            'ssp370': 'SSP3-7.0',
            'ssp585': 'SSP5-8.5',
        }
        self.map_method = {
            'ar6_orig': '#0 EBM-ε AR6 orig',
            'ar6': '#1 EBM-ε AR6',
            's21': '#2 EBM-ε S21',
            'mce-2l': '#3 MCE-2l',
        }
        self.map_color = {
            'Constrained CMIP6': 'C7',
            'ECS-TCR mapped emulator': 'C6',
            '#0 EBM-ε AR6 orig': 'C9',
            '#1 EBM-ε AR6': 'C0',
            '#2 EBM-ε S21': 'C2',
            '#3 MCE-2l': 'C1',
            'Reference': 'C3',
        }

    def plot_quantile_range(self, dfin, axes=None, **kw):
        """Make quantile range plot

        Parameters
        ----------
        dfin
            Input DataFrame indexed with 'central', 'likely__lower/upper',
            and 'very_likely__lower/upper' on axis 0, and with two or three
            levels on axis 1. The axis 1 levels are the parameters, groups,
            and members in order, and a dummy group will be created in the
            case of two-level axis.
        axes, optional
            An array-like of Axes object, by default None, in which case
            created internally.
        kw, optional control parameters as follows
            parm_order
            group_order
            member_order
            map_color
            map_name_unit
            shrink
            kw_space
            col
            kw_legend
        """
        nlevels = dfin.columns.nlevels

        if nlevels == 2:
            dfin = pd.concat({'dummy': dfin}, axis=1).reorder_levels([1, 0, 2], axis=1)

        mi = dfin.columns.remove_unused_levels()

        parm_order = kw.get('parm_order', mi.levels[0].tolist())
        group_order = kw.get('group_order', mi.levels[1].tolist())
        member_order = kw.get('member_order', mi.levels[2].tolist())
        map_color = kw.get(
            'map_color',
            {k: f'C{i}' for i, k in enumerate(member_order)},
        )
        map_name_unit = kw.get('map_name_unit', {})
        shrink = kw.get('shrink', 0.7)

        kw_space = kw.get(
            'kw_space',
            {'height': 1.5, 'aspect': 2.5, 'wspace': 1.2},
        )
        col = kw.get('col', 1)

        nparms = len(parm_order)
        ngroups = len(group_order)
        nmembers = len(member_order)
        colors = [map_color[member] for member in member_order] * ngroups

        yvals = np.arange(ngroups * nmembers)[::-1].reshape((-1, nmembers)) + 0.5
        ym = yvals.mean(axis=1)
        yvals = ((yvals - ym[:, None]) * shrink + ym[:, None]).ravel()

        if axes is None:
            if 'extend' in kw_space:
                self.init_general(**kw_space)
            else:
                self.init_regular(nparms, col, kw_space=kw_space)

            axes = self()

        mi = pd.MultiIndex.from_product([group_order, member_order])

        for ax, pn in zip(axes, parm_order):
            df = dfin[pn].reindex(columns=mi)
            ax.hlines(
                yvals, df.loc['very_likely__lower'], df.loc['very_likely__upper'],
                color=colors, lw=1., zorder=1,
            )
            ax.hlines(
                yvals, df.loc['likely__lower'], df.loc['likely__upper'],
                color=colors, lw=4., zorder=1,
            )
            ax.scatter(
                df.loc['central'], yvals,
                marker='o', facecolor='w', edgecolors=colors,
            )
            ax.set_yticks(ym)
            if nlevels == 3:
                ax.set_yticklabels(group_order)
                ax.tick_params(axis='y', labelleft=True, left=False)
            else:
                ax.tick_params(axis='y', labelleft=False, left=False)

            ax.set_ylim(0, ngroups*nmembers)
            ax.spines['left'].set_visible(False)

            name, unit = map_name_unit.get(pn, (pn, ''))
            if unit != '':
                ax.set_xlabel(f'{name} ({unit})')
            else:
                ax.set_xlabel(name)

            ax.grid(axis='x')

        handles = [
            mpl.lines.Line2D([0], [0], color=color, lw=1.5)
            for color in colors[:nmembers]
        ] + [
            mpl.patches.Patch(alpha=0, linewidth=0),
            mpl.lines.Line2D([0], [0], ls='None', marker='o', mec='k', mfc='w'),
            mpl.lines.Line2D([0], [0], color='k', lw=3., solid_capstyle='butt'),
            mpl.lines.Line2D([0], [0], color='k', lw=1., ls='-'),
        ]
        labels = member_order + ['', 'Central', 'likely (66%)', 'very likely (90%)']

        kw_legend = kw.get('kw_legend', {
            'loc': 'upper left',
            'bbox_to_anchor': (1.07, 0.98),
        })
        kw_legend['bbox_to_anchor'] = self.get_fig_position_relto_axes(
            kw_legend['bbox_to_anchor'],
        )
        self.figure.legend(handles, labels, **kw_legend)
