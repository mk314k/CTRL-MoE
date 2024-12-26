from sumo_rl import SumoEnvironment
from .contexts import Context

class SUMOContextualEnv(SumoEnvironment):
    def reset_context(self, context:Context):
        self.context = context
        self.change_reward_weight(**context.reward_weights())
        
    def change_reward_weight(self, **kwargs):
        self.pressure_weight = kwargs.get('pressure', 0)
        self.avg_speed_weight = kwargs.get('avg_speed', 0)
        self.diff_wait_weight = kwargs.get('diff_wait_time', 1)
        self.total_queue_weight = kwargs.get('queue', -1)

    def _compute_rewards(self):
        self.rewards.update(
            {
                ts: self._process_ts_reward(ts)
                for ts in self.ts_ids
                if self.traffic_signals[ts].time_to_act or self.fixed_ts
            }
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act or self.fixed_ts}
    
    def _process_ts_reward(self, ts):
        reward = self.pressure_weight * self.traffic_signals[ts].get_pressure()
        reward += self.avg_speed_weight * self.traffic_signals[ts].get_average_speed()
        reward += self.total_queue_weight * self.traffic_signals[ts].get_total_queued()
        if self.diff_wait_weight != 0:
            ts_wait = sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) / 100.0
            reward += self.diff_wait_weight * (self.traffic_signals[ts].last_measure - ts_wait)
            self.traffic_signals[ts].last_measure = ts_wait
        reward = reward / 100
        self.traffic_signals[ts].last_reward = reward
        return reward
