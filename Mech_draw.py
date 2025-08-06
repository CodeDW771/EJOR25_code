import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def draw_task1d(data_table):
    
    demands = data_table['d_value']
    sp_values = data_table['Ind_SP']
    vcg_values = data_table['Ind_VCG']
        
    nodes = ['N0', 'N1', 'N2', 'N3']
    
    # Set plot parameters
    bar_width = 0.35  
    n_demands = len(demands)
    n_nodes = len(nodes)
    total_positions = n_demands * n_nodes
    indices = np.arange(total_positions)  

    fig, ax = plt.subplots(figsize=(15, 5))  
    plt.subplots_adjust(bottom=0.25)  
    

    sp_color = '#E9B5C1'  # color for SP
    vcg_color = '#B4C5D9'   # color for VCG
    
    # Data
    sp_flat = []
    vcg_flat = []
    demand_labels = []
    node_labels = []
    
    for i, demand in enumerate(demands):
        for j, node in enumerate(nodes):
            sp_flat.append(sp_values[i][j])
            vcg_flat.append(vcg_values[i][j])
            demand_labels.append(f'{demand}d')
            node_labels.append(node)
    
    sp_pos = indices - bar_width/2
    vcg_pos = indices + bar_width/2

    for i in range(1, n_demands):
        pos = i * n_nodes - 0.5
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.7)

    ax.bar(sp_pos, sp_flat, width=bar_width, color=sp_color, 
           edgecolor='black', label='SP', zorder=3)
    
    ax.bar(vcg_pos, vcg_flat, width=bar_width, color=vcg_color, 
           edgecolor='black', hatch='//', label='VCG', zorder=3)
    
    max_val = max(max(sp_flat), max(vcg_flat))
    min_spacing = max_val * 0.05 
    base_offset = min(500, max_val * 0.05) 

    def format_label(val):
        """Format large numbers in 'K' notation with 2 decimal places."""
        if val >= 1000:
            s = f'{val/1000:.2f}K'

            if s.endswith('.00K'):
                return s.rstrip('00').rstrip('.') + 'K'
            elif s.endswith('0K'):
                return s.rstrip('0K').rstrip('.') + 'K'
            return s
        return f'{val:.0f}'
    
    placed_labels = []
    
    for i, (sp_val, vcg_val) in enumerate(zip(sp_flat, vcg_flat)):
        font_props = {'fontfamily': 'Times New Roman', 'fontsize': 10}

        sp_y = sp_val + base_offset

        for prev_x, prev_y in placed_labels:
            if abs(sp_pos[i] - prev_x) < 0.5: 
                if abs(sp_y - prev_y) < min_spacing:
                    sp_y = prev_y + min_spacing
        ax.text(sp_pos[i], sp_y, format_label(sp_val), 
                ha='center', va='bottom', **font_props)
        placed_labels.append((sp_pos[i], sp_y))

        vcg_y = vcg_val + base_offset

        if vcg_y < sp_y + min_spacing/2:
            vcg_y = sp_y + min_spacing
        ax.text(vcg_pos[i], vcg_y, format_label(vcg_val), 
                ha='center', va='bottom', **font_props)
        placed_labels.append((vcg_pos[i], vcg_y))
    

    ax.set_xticks(indices)
    ax.set_xticklabels(node_labels, fontsize=14)
    

    demand_label_positions = [indices[i * n_nodes + n_nodes//2] for i in range(n_demands)]
    for i, pos in enumerate(demand_label_positions):
        ax.text(pos, -0.13 * max(sp_flat + vcg_flat), demand_labels[i * n_nodes], 
                ha='center', va='top', fontsize=16, fontweight='bold')
    
    
    
    # Set other attributes
    ax.set_ylabel(r'Incentive $\Pi_i$', fontsize=16)
    #ax.set_title('SP and VCG Values by Demand and Node', fontsize=14)
    ax.set_ylim(0, max(max(sp_flat), max(vcg_flat)) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7,zorder=0)
    ax.legend(loc='upper right', fontsize=16)
    
    plt.rcParams.update({
        "mathtext.fontset": 'cm', 
        "font.family": 'serif'
    })

    def thousands_formatter(x, pos):
        """Format large numbers in 'K' notation with 1 decimal places."""
        if abs(x) >= 1000:
            formatted = f'{x/1000:.1f}K'
            if formatted.endswith('.0K'):
                return formatted.replace('.0K', 'K')
            return formatted
        else:
            return f'{x:.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.tick_params(axis='y', labelsize=14) 
    
    #fig.text(0.5, 0.05, 'Nodes', ha='center', va='center', fontsize=12)
    #fig.text(0.5, 0.01, 'Demand Groups', ha='center', va='center', fontsize=12, fontweight='bold')
    
    
    return fig


def draw_task1m(data_table):
    
    demands = data_table['m_value']
    sp_values = data_table['Ind_SP']
    vcg_values = data_table['Ind_VCG']
        
    nodes = ['N0', 'N1', 'N2', 'N3']
    
    # Set plot parameters
    bar_width = 0.35  
    n_demands = len(demands)
    n_nodes = len(nodes)
    total_positions = n_demands * n_nodes
    indices = np.arange(total_positions)  
    
    fig, ax = plt.subplots(figsize=(15, 6))  
    plt.subplots_adjust(bottom=0.25)  

    sp_color = '#E9B5C1'  # color for SP
    vcg_color = '#B4C5D9'   # color for VCG
    
    # Data
    sp_flat = []
    vcg_flat = []
    demand_labels = []
    node_labels = []
    
    for i, demand in enumerate(demands):
        for j, node in enumerate(nodes):
            sp_flat.append(sp_values[i][j])
            vcg_flat.append(vcg_values[i][j])
            demand_labels.append(f'{demand}m')
            node_labels.append(node)
    
    sp_pos = indices - bar_width/2
    vcg_pos = indices + bar_width/2
    
    for i in range(1, n_demands):
        pos = i * n_nodes - 0.5
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.7)
    
    # SP bar
    ax.bar(sp_pos, sp_flat, width=bar_width, color=sp_color, 
           edgecolor='black', label='SP', zorder=3)
    
    # VCG bar
    ax.bar(vcg_pos, vcg_flat, width=bar_width, color=vcg_color, 
           edgecolor='black', hatch='//', label='VCG', zorder=3)
    
    max_val = max(max(sp_flat), max(vcg_flat))
    min_spacing = max_val * 0.05 
    base_offset = min(500, max_val * 0.05) 
    
    def format_label(val):
        """Format large numbers in 'K' notation with 2 decimal places."""
        if val >= 1000:
            s = f'{val/1000:.2f}K'

            if s.endswith('.00K'):
                return s.rstrip('00').rstrip('.') + 'K'
            elif s.endswith('0K'):
                return s.rstrip('0K').rstrip('.') + 'K'
            return s
        return f'{val:.0f}'
    
    placed_labels = []
    
    for i, (sp_val, vcg_val) in enumerate(zip(sp_flat, vcg_flat)):
        font_props = {'fontfamily': 'Times New Roman', 'fontsize': 10}
        

        sp_y = sp_val + base_offset

        for prev_x, prev_y in placed_labels:
            if abs(sp_pos[i] - prev_x) < 0.5: 
                if abs(sp_y - prev_y) < min_spacing:
                    sp_y = prev_y + min_spacing
        ax.text(sp_pos[i], sp_y, format_label(sp_val), 
                ha='center', va='bottom', **font_props)
        placed_labels.append((sp_pos[i], sp_y))
        

        vcg_y = vcg_val + base_offset

        if vcg_y < sp_y + min_spacing/2:
            vcg_y = sp_y + min_spacing
        ax.text(vcg_pos[i], vcg_y, format_label(vcg_val), 
                ha='center', va='bottom', **font_props)
        placed_labels.append((vcg_pos[i], vcg_y))
    

    ax.set_xticks(indices)
    ax.set_xticklabels(node_labels, fontsize=14)
    

    demand_label_positions = [indices[i * n_nodes + n_nodes//2] for i in range(n_demands)]
    for i, pos in enumerate(demand_label_positions):
        ax.text(pos, -0.13 * max(sp_flat + vcg_flat), demand_labels[i * n_nodes], 
                ha='center', va='top', fontsize=16, fontweight='bold')
    
    
    
    # Set other attributes
    ax.set_ylabel(r'Incentive $\Pi_i$', fontsize=16)
    # ax.set_title('SP and VCG Values by Demand and Node', fontsize=14)
    ax.set_ylim(0, max(max(sp_flat), max(vcg_flat)) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7,zorder=0)
    ax.legend(loc='upper right', fontsize=16)
    
    plt.rcParams.update({
        "mathtext.fontset": 'cm', 
        "font.family": 'serif'
    })
    

    def thousands_formatter(x, pos):
        """Format large numbers in 'K' notation with 1 decimal places."""
        if abs(x) >= 1000:
            formatted = f'{x/1000:.1f}K'
            if formatted.endswith('.0K'):
                return formatted.replace('.0K', 'K')
            return formatted
        else:
            return f'{x:.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.tick_params(axis='y', labelsize=14) 
    
    #fig.text(0.5, 0.05, 'Nodes', ha='center', va='center', fontsize=12)
    #fig.text(0.5, 0.01, 'Demand Groups', ha='center', va='center', fontsize=12, fontweight='bold')
    
    
    return fig


def draw_task1c(data_table):
    
    demands = data_table['c_value']
    sp_values = data_table['Ind_SP']
    vcg_values = data_table['Ind_VCG']
        
    nodes = ['N0', 'N1', 'N2', 'N3']
    
    # Set plot parameters
    bar_width = 0.35
    n_demands = len(demands)
    n_nodes = len(nodes)
    total_positions = n_demands * n_nodes
    indices = np.arange(total_positions) 
    
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.subplots_adjust(bottom=0.25)
    
    # Color
    sp_color = '#E9B5C1'
    vcg_color = '#B4C5D9'
    
    # Data
    sp_flat = []
    vcg_flat = []
    demand_labels = []
    node_labels = []
    
    for i, demand in enumerate(demands):
        for j, node in enumerate(nodes):
            sp_flat.append(sp_values[i][j])
            vcg_flat.append(vcg_values[i][j])
            demand_labels.append(f'{demand}c')
            node_labels.append(node)
    
    # Position of bar
    sp_pos = indices - bar_width/2
    vcg_pos = indices + bar_width/2
    
    for i in range(1, n_demands):
        pos = i * n_nodes - 0.5
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.7)
    
    # SP bar
    ax.bar(sp_pos, sp_flat, width=bar_width, color=sp_color, 
           edgecolor='black', label='SP', zorder=3)
    
    # VCG bar
    ax.bar(vcg_pos, vcg_flat, width=bar_width, color=vcg_color, 
           edgecolor='black', hatch='//', label='VCG', zorder=3)
    
    max_val = max(max(sp_flat), max(vcg_flat))
    min_spacing = max_val * 0.05
    base_offset = min(500, max_val * 0.05)
    
    def format_label(val):
        """Format large numbers in 'K' notation with 2 decimal places."""
        if val >= 1000:
            s = f'{val/1000:.2f}K'
            # 移除不必要的零
            if s.endswith('.00K'):
                return s.rstrip('00').rstrip('.') + 'K'
            elif s.endswith('0K'):
                return s.rstrip('0K').rstrip('.') + 'K'
            return s
        return f'{val:.0f}'
    
    placed_labels = []
    
    for i, (sp_val, vcg_val) in enumerate(zip(sp_flat, vcg_flat)):
        font_props = {'fontfamily': 'Times New Roman', 'fontsize': 10}
        # SP part
        sp_y = sp_val + base_offset

        for prev_x, prev_y in placed_labels:
            if abs(sp_pos[i] - prev_x) < 0.5:
                if abs(sp_y - prev_y) < min_spacing:
                    sp_y = prev_y + min_spacing
        ax.text(sp_pos[i], sp_y, format_label(sp_val), 
                ha='center', va='bottom', **font_props)
        placed_labels.append((sp_pos[i], sp_y))
        
        # VCG part
        vcg_y = vcg_val + base_offset

        if vcg_y < sp_y + min_spacing/2:
            vcg_y = sp_y + min_spacing
        ax.text(vcg_pos[i], vcg_y, format_label(vcg_val), 
                ha='center', va='bottom', **font_props)
        placed_labels.append((vcg_pos[i], vcg_y))
    

    ax.set_xticks(indices)
    ax.set_xticklabels(node_labels, fontsize=14)
    

    demand_label_positions = [indices[i * n_nodes + n_nodes//2] for i in range(n_demands)]
    for i, pos in enumerate(demand_label_positions):
        ax.text(pos, -0.13 * max(sp_flat + vcg_flat), demand_labels[i * n_nodes], 
                ha='center', va='top', fontsize=16, fontweight='bold')
    
    

    ax.set_ylabel(r'Incentive $\Pi_i$', fontsize=16)
    #ax.set_title('SP and VCG Values by Demand and Node', fontsize=14)
    ax.set_ylim(0, max(max(sp_flat), max(vcg_flat)) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7,zorder=0)
    ax.legend(loc='upper right', fontsize=16)
    
    plt.rcParams.update({
        "mathtext.fontset": 'cm',
        "font.family": 'serif'
    })
    

    def thousands_formatter(x, pos):
        """Format large numbers in 'K' notation with 1 decimal places."""
        if abs(x) >= 1000:
            formatted = f'{x/1000:.1f}K'
            if formatted.endswith('.0K'):
                return formatted.replace('.0K', 'K')
            return formatted
        else:
            return f'{x:.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.tick_params(axis='y', labelsize=14) 
    

    #fig.text(0.5, 0.05, 'Nodes', ha='center', va='center', fontsize=12)
    #fig.text(0.5, 0.01, 'Demand Groups', ha='center', va='center', fontsize=12, fontweight='bold')
    
    
    return fig

def draw_task2_0(data_table):
    
    
    delta = data_table['Delta']
    n_data_list = []
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    colors = [ '#DD847E', '#E9D389', '#A7D398', '#74A3D4']
    
    for i in range(len(data_table)-1):
        key = f'N{i}'
        n_data = data_table[key]
        n_data_list.append(n_data)
        
        plt.plot(delta, n_data_list[i], 
                 label=f'N{i}',
                 color=colors[i],
                 linewidth=3.5)
    
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5 )
    
    ax.set_xlim(min(delta)-0.5, max(delta)+0.5)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_xticklabels(np.arange(-5, 6, 1),fontsize=14)
    
    plt.xlabel(r'$\Delta_0$', fontsize=16)
    plt.ylabel(r'Profit $|u_i|$', fontsize=16)
    
    #plt.title('Delta vs N Values', fontsize=16)
    plt.legend(fontsize=16)
    
    plt.grid(True, alpha=0.3)
    
    
    def thousands_formatter(x, pos):
        """Format large numbers in 'K' notation with 1 decimal places."""
        if abs(x) >= 1000:
            formatted = f'{x/1000:.1f}K'
            if formatted.endswith('.0K'):
                return formatted.replace('.0K', 'K')
            return formatted
        else:
            return f'{x:.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_ylim(0, 6300)
    ax.tick_params(axis='y', labelsize=14) 
    
    plt.tight_layout()
    
    
    return fig

def draw_task2_1(data_table):
    
    
    delta = data_table['Delta']
    n_data_list = []
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    colors = [ '#DD847E', '#E9D389', '#A7D398', '#74A3D4']
    
    for i in range(len(data_table)-1):
        key = f'N{i}'
        n_data = data_table[key]
        n_data_list.append(n_data)
        
        plt.plot(delta, n_data_list[i], 
                 label=f'N{i}',
                 color=colors[i],
                 linewidth=3.5)
    
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5 )
    
    ax.set_xlim(min(delta)-0.5, max(delta)+0.5)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_xticklabels(np.arange(-5, 6, 1),fontsize=14)
    

    plt.xlabel(r'$\Delta_1$', fontsize=16)
    plt.ylabel(r'Profit $|u_i|$', fontsize=16)

    #plt.title('Delta vs N Values', fontsize=16)
    plt.legend(fontsize=16)
    
    plt.grid(True, alpha=0.3)
    
    
    def thousands_formatter(x, pos):
        """Format large numbers in 'K' notation with 1 decimal places."""
        if abs(x) >= 1000:
            formatted = f'{x/1000:.1f}K'
            if formatted.endswith('.0K'):
                return formatted.replace('.0K', 'K')
            return formatted
        else:
            return f'{x:.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_ylim(0, 6300)
    ax.tick_params(axis='y', labelsize=14) 
    
    plt.tight_layout()
    
    
    return fig

def draw_task2_2(data_table):
    
    
    delta = data_table['Delta']
    n_data_list = []
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    colors = [ '#DD847E', '#E9D389', '#A7D398', '#74A3D4']
    
    for i in range(len(data_table)-1):
        key = f'N{i}'
        n_data = data_table[key]
        n_data_list.append(n_data)
        
        plt.plot(delta, n_data_list[i], 
                 label=f'N{i}',
                 color=colors[i],
                 linewidth=3.5)
    
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5 )
    
    ax.set_xlim(min(delta)-0.5, max(delta)+0.5)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_xticklabels(np.arange(-5, 6, 1),fontsize=14)
    
    plt.xlabel(r'$\Delta_2$', fontsize=16)
    plt.ylabel(r'Profit $|u_i|$', fontsize=16)

    #plt.title('Delta vs N Values', fontsize=16)
    plt.legend(fontsize=16)
    
    plt.grid(True, alpha=0.3)
    
    
    def thousands_formatter(x, pos):
        """Format large numbers in 'K' notation with 1 decimal places."""
        if abs(x) >= 1000:
            formatted = f'{x/1000:.1f}K'
            if formatted.endswith('.0K'):
                return formatted.replace('.0K', 'K')
            return formatted
        else:
            return f'{x:.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_ylim(0, 6300)
    ax.tick_params(axis='y', labelsize=14) 
    
    plt.tight_layout()
    
    
    return fig

def draw_task2_3(data_table):
    
    
    delta = data_table['Delta']
    n_data_list = []
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    colors = [ '#DD847E', '#E9D389', '#A7D398', '#74A3D4']
    
    for i in range(len(data_table)-1):
        key = f'N{i}'
        n_data = data_table[key]
        n_data_list.append(n_data)
        
        plt.plot(delta, n_data_list[i], 
                 label=f'N{i}',
                 color=colors[i],
                 linewidth=3.5)
    
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5 )
    
    ax.set_xlim(min(delta)-0.5, max(delta)+0.5)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_xticklabels(np.arange(-5, 6, 1),fontsize=14)

    plt.xlabel(r'$\Delta_3$', fontsize=16)
    plt.ylabel(r'Profit $|u_i|$', fontsize=16)

    #plt.title('Delta vs N Values', fontsize=16)
    plt.legend(fontsize=16)

    plt.grid(True, alpha=0.3)
    
    
    def thousands_formatter(x, pos):
        """Format large numbers in 'K' notation with 1 decimal places."""
        if abs(x) >= 1000:
            formatted = f'{x/1000:.1f}K'
            if formatted.endswith('.0K'):
                return formatted.replace('.0K', 'K')
            return formatted
        else:
            return f'{x:.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_ylim(0, 6300)
    ax.tick_params(axis='y', labelsize=14) 
    
    plt.tight_layout()
    
    
    return fig



def draw_task3(data_table, optimal_base):
    
    
    case = data_table['Case']
    n_data_list = []
        
    fig = plt.figure(figsize=(16, 5.5))
    ax = fig.add_subplot(111)
    
    colors = [ '#DD847E', '#E9D389', '#A7D398', '#74A3D4']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i in range(len(data_table)-1):
        key = f'N{i}'
        n_data = data_table[key]
        n_data_list.append(n_data)
        
        plt.plot(case, n_data_list[i], 
                 label=f'N{i}',
                 linestyle='--',
                 color=colors[i],
                 marker=markers[i],
                 markersize = 12,
                 linewidth=3.5)
        
        ax.axhline( y = optimal_base[i], 
                   color=colors[i], 
                   linestyle=':',
                   linewidth=3)
          
    
    ax.set_xticks(range(len(case)))
    ax.set_xticklabels(case, rotation=45, ha='right',fontsize=14)
    #ax.set_xticklabels(case, fontsize=12)
    ax.set_xlabel('Case', fontsize=16)
    ax.set_ylabel(r'Profit $|u_i|$', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    ax.legend(unique_handles, unique_labels, 
              bbox_to_anchor=(1.005, 1), fontsize=16,
              loc='upper left')
    
    def thousands_formatter(x, pos):
        if abs(x) >= 1000:
            formatted = f'{x/1000:.1f}K'
            if formatted.endswith('.0K'):
                return formatted.replace('.0K', 'K')
            return formatted
        else:
            return f'{x:.0f}'
        
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.tick_params(axis='y', labelsize=14) 
    plt.tight_layout()
    #plt.show()
    
    
    return fig


