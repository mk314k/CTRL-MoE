from .contexts import Context
from .sumoenv import SUMOContextualEnv

base_path = './nets/2way-single-intersection/single-intersection'
def make_context_env(ctx, reset=False):
    ctx_name = '_'.join(ctx)
    context = Context(ctx_name, f'{base_path}-{ctx_name}.rou.xml', reset=reset)
    sumo_env = SUMOContextualEnv(
        net_file=base_path+'.net.xml',
        route_file=context.rou_path,
        out_csv_name=f'./outputs/{ctx_name}/out', 
        single_agent = True,
        use_gui=False,
        num_seconds=1000
    )
    sumo_env.reset_context(context)
    return context, sumo_env