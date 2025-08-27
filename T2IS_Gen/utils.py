"""
Utility functions for T2IS generation and image processing.
"""

def calculate_layout_dimensions(num_prompts, sub_height, sub_width):
    """Calculate total height and width based on number of prompts and sub-image dimensions.
    
    Args:
        num_prompts (int): Number of sub-images/prompts
        sub_height (int): Height of each sub-image
        sub_width (int): Width of each sub-image
        
    Returns:
        tuple: Total (height, width) dimensions
    """
    if num_prompts == 4:
        # 2x2 layout
        height = sub_height * 2
        width = sub_width * 2
    elif num_prompts == 6:
        # 2x3 layout
        height = sub_height * 2
        width = sub_width * 3
    elif num_prompts == 8:
        # 2x4 layout
        height = sub_height * 2
        width = sub_width * 4
    elif num_prompts == 9:
        # 3x3 layout
        height = sub_height * 3
        width = sub_width * 3
    else:
        # Default: 1xN layout
        height = sub_height
        width = sub_width * num_prompts
    return height, width


def calculate_cutting_layout(num_prompts):
    """Calculate rows and columns for cutting images based on number of prompts.
    
    Args:
        num_prompts (int): Number of sub-images/prompts
        
    Returns:
        tuple: (rows, cols) for cutting layout
    """
    if num_prompts == 4:
        # 2*2 切割
        rows, cols = 2, 2
    elif num_prompts == 6:
        # 2*3 切割
        rows, cols = 2, 3
    elif num_prompts == 8:
        # 2*4 切割
        rows, cols = 2, 4
    elif num_prompts == 9:
        # 3*3 切割
        rows, cols = 3, 3
    else:
        # 默认沿着宽切割
        rows, cols = 1, num_prompts
    
    return rows, cols


def get_grid_params(size):
    """Calculate grid parameters for image layout based on size.
    
    Args:
        size (int): Number of sub-images/prompts
        
    Returns:
        tuple: (m_offsets, n_offsets, m_scales, n_scales) for grid layout
    """
    if size == 4:
        rows, cols = 2, 2
    elif size == 6:
        rows, cols = 2, 3  
    elif size == 8:
        rows, cols = 2, 4
    elif size == 9:
        rows, cols = 3, 3
    else:
        rows, cols = 1, size

    m_scale = 1.0/cols
    n_scale = 1.0/rows
    
    m_offsets = []
    n_offsets = []
    m_scales = []
    n_scales = []
    
    for row in range(rows):
        for col in range(cols):
            m_offsets.append(col * m_scale)
            n_offsets.append(row * n_scale)
            m_scales.append(m_scale)
            n_scales.append(n_scale)
            
    return m_offsets, n_offsets, m_scales, n_scales



