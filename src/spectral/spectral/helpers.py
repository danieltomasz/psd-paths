def style_cfg(*, node, node_class): 
    """
    Returns: (style_dict, node_class, legend_label)
    """
    tag_value = node.tags.get("kind")

    # Style 1: Visualization (Gold)
    if tag_value == "visualization":
        return ({"style": "filled", "fillcolor": "#FFD700"}, node_class, "Plotting")
    
    # Style 2: Report (Pale Green) <--- NEW SECTION
    elif tag_value == "report":
        return ({"style": "filled", "fillcolor": "#98FB98"}, node_class, "Report")
    
    # Default: Data Processing (Blue/Grey default)
    return ({}, node_class, None)
