import os
import random
import torch

context_weights = {
        "morning": {"pressure": 0.3, "avg_speed": 0.4, "queue": -0.3, "diff_wait_time": 0.2},
        "midday": {"pressure": 0.2, "avg_speed": 0.5, "queue": -0.1, "diff_wait_time": 0.1},
        "evening": {"pressure": 0.2, "avg_speed": 0.3, "queue": -0.3, "diff_wait_time": 0.2},
        "night": {"pressure": 0.1, "avg_speed": 0.6, "queue": -0.1, "diff_wait_time": 0.1},
        "rainy": {"pressure": 0.4, "avg_speed": 0.3, "queue": -0.2, "diff_wait_time": 0.1},
        "warm": {"pressure": 0.2, "avg_speed": 0.4, "queue": -0.2, "diff_wait_time": 0.2},
        "windy": {"pressure": 0.3, "avg_speed": 0.3, "queue": -0.3, "diff_wait_time": 0.1},
        "hew": {"pressure": 0.4, "avg_speed": 0.2, "queue": -0.2, "diff_wait_time": 0.1},
        "holiday": {"pressure": 0.1, "avg_speed": 0.5, "queue": -0.3, "diff_wait_time": 0.1},
        "workday": {"pressure": 0.2, "avg_speed": 0.3, "queue": -0.3, "diff_wait_time": 0.2},
    }

base_flow = {
    # Time-based contexts
    "morning": {"vehsPerHour": 600, "adjustment": 1.0},  # Morning has high traffic
    "midday": {"vehsPerHour": 400, "adjustment": 0.8},   # Midday has moderate traffic
    "evening": {"vehsPerHour": 700, "adjustment": 1.1},  # Evening has the highest traffic
    "night": {"vehsPerHour": 200, "adjustment": 0.5},    # Night has low traffic

    # Weather-based contexts
    "rainy": {"adjustment": 0.8},                       # Traffic flow decreases in rain
    "warm": {"adjustment": 1.05},                        # Normal traffic flow in warm weather
    "windy": {"adjustment": 0.7},                       # Slightly reduced flow in windy weather
    "hew": {"adjustment": 0.6},                         # Reduced flow in hew (humid and windy)

    # Day-based contexts
    "holiday": {"adjustment": 0.6},                     # Traffic significantly reduces on holidays
    "workday": {"adjustment": 1.4},                     # Higher traffic on workdays
}


class Context:
    def __init__(self, context_name, rou_path, alpha=0.1, reset=False):
        """
        Initialize a context with its name and rou.xml path.

        Args:
            context_name (str or list of str): The name(s) of the context (e.g., 'morning', 'rainy').
            rou_path (str): Path to the generated .rou.xml file.
            alpha (float): Degree of randomness for route generation.
        """
        self.name = context_name if isinstance(context_name, list) else [context_name]
        self.rou_path = rou_path
        self.alpha = alpha

        if (not os.path.exists(self.rou_path)) or reset:
            print(f"File {self.rou_path} not found. Generating...")
            self.generate_rou_file(self.name, self.rou_path, alpha=self.alpha)

    def reward_weights(self):
        """
        Outputs the weights for the reward function based on the context.

        Returns:
            dict: Averaged weights for the given context(s).
        """

        # Default weights for unknown contexts
        default_weights = {"pressure": 0, "avg_speed": 0, "queue": 0, "diff_wait_time": 1}

        # Calculate averaged weights for multiple contexts
        combined_weights = default_weights.copy()
        for ctx in self.name:
            weights = context_weights.get(ctx, default_weights)
            for key in combined_weights:
                combined_weights[key] += weights[key]

        # Average weights over the number of contexts
        num_contexts = len(self.name)
        averaged_weights = {key: value / num_contexts for key, value in combined_weights.items()}
        return averaged_weights
    
    def as_tensor(self, device='cpu'):
        return torch.tensor([*list(self.reward_weights().values()), self.get_flow_rate(self.name)]).view(1, -1).to(device)
    
    @staticmethod
    def get_flow_rate(ctx):
        if isinstance(ctx, list):
            vehs_per_hour = 400
            adjustment = 1
            for c in ctx:
                if c in base_flow:
                    vehs_per_hour = max(base_flow[c].get("vehsPerHour", 400), vehs_per_hour)
                    adjustment = min(base_flow[c].get("adjustment", 1.0), adjustment)
        elif ctx in base_flow:
            vehs_per_hour = base_flow[ctx].get("vehsPerHour", 400)
            adjustment = base_flow[ctx].get("adjustment", 1.0)

        return int(vehs_per_hour * adjustment)
    
    @staticmethod
    def generate_rou_file(ctx, output_path="context_rou.xml", alpha=0.05):
        """
        Generates a .rou.xml file with flows across all available routes, introducing randomness.

        Args:
            context (str or list of str): Context(s) like 'rainy', 'morning', 'workday'.
            output_path (str): Path to save the generated .rou.xml file.
            alpha (float): Degree of randomness (0 for no randomness, up to 1 for high randomness).

        Returns:
            str: Path to the generated .rou.xml file.
        """
        # Base routes for 2 way single intersection
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

        # Default settings
        start = 0
        end = 3600

        base_flow_rate = Context.get_flow_rate(ctx)

        with open(output_path, "w") as file:
            file.write("<routes>\n")

            # Adding routes
            for route in routes:
                file.write(f'    <route id="{route["id"]}" edges="{route["edges"]}"/>\n')

            # Adding flows with randomness
            for route in routes:
                random_factor = 1 + random.uniform(-alpha, alpha)  # Scaling by (1 ± alpha)
                flow_rate = int(base_flow_rate * random_factor)
                file.write(
                    f'    <flow id="flow_{route["id"]}" route="{route["id"]}" '
                    f'begin="{start}" end="{end}" vehsPerHour="{flow_rate}" '
                    f'departSpeed="max" departLane="best"/>\n'
                )

            file.write("</routes>\n")

        return output_path