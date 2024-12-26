import random

def generate_rou_file(context, output_path="context_rou.xml", alpha=0.1):
    """
    Generates a .rou.xml file with flows across all available routes, introducing randomness.

    Args:
        context (str or list of str): Context(s) like 'rainy', 'morning', 'workday'.
        output_path (str): Path to save the generated .rou.xml file.
        alpha (float): Degree of randomness (0 for no randomness, up to 1 for high randomness).

    Returns:
        str: Path to the generated .rou.xml file.
    """
    # Define base routes
    routes = [
        {"id": "route_ns", "edges": "n_t t_s"},
        {"id": "route_nw", "edges": "n_t t_w"},
        {"id": "route_ne", "edges": "n_t t_e"},
        {"id": "route_we", "edges": "w_t t_e"},
        {"id": "route_wn", "edges": "w_t t_n"},
        {"id": "route_ws", "edges": "w_t t_s"},
        {"id": "route_ew", "edges": "e_t t_w"},
        {"id": "route_en", "edges": "e_t t_n"},
        {"id": "route_es", "edges": "e_t t_s"},
        {"id": "route_sn", "edges": "s_t t_n"},
        {"id": "route_se", "edges": "s_t t_e"},
        {"id": "route_sw", "edges": "s_t t_w"},
    ]

    # Base flow parameters
    base_flow = {
        "morning": {"vehsPerHour": 500, "adjustment": 1.0},
        "evening": {"vehsPerHour": 600, "adjustment": 1.0},
        "rainy": {"adjustment": 0.8},  # Reduce flow in rain
        "holiday": {"adjustment": 0.6},  # Fewer vehicles on holidays
        "workday": {"adjustment": 1.2},  # More vehicles on workdays
    }

    # Default settings
    vehs_per_hour = 400
    adjustment = 1.0
    start = 0
    end = 7200

    # Apply context adjustments
    if isinstance(context, list):
        for ctx in context:
            if ctx in base_flow:
                if "vehsPerHour" in base_flow[ctx]:
                    vehs_per_hour = base_flow[ctx]["vehsPerHour"]
                if "adjustment" in base_flow[ctx]:
                    adjustment *= base_flow[ctx]["adjustment"]
    elif context in base_flow:
        if "vehsPerHour" in base_flow[context]:
            vehs_per_hour = base_flow[context]["vehsPerHour"]
        if "adjustment" in base_flow[context]:
            adjustment *= base_flow[context]["adjustment"]

    # Adjust base flow rate
    base_flow_rate = int(vehs_per_hour * adjustment)

    # Write .rou.xml file
    with open(output_path, "w") as file:
        file.write("<routes>\n")

        # Add routes
        for route in routes:
            file.write(f'    <route id="{route["id"]}" edges="{route["edges"]}"/>\n')

        # Add flows with randomness
        for route in routes:
            # Introduce randomness to the flow rate
            random_factor = 1 + random.uniform(-alpha, alpha)  # Scale by (1 ± alpha)
            flow_rate = int(base_flow_rate * random_factor)
            file.write(
                f'    <flow id="flow_{route["id"]}" route="{route["id"]}" '
                f'begin="{start}" end="{end}" vehsPerHour="{flow_rate}" '
                f'departSpeed="max" departLane="best"/>\n'
            )

        file.write("</routes>\n")

    return output_path
