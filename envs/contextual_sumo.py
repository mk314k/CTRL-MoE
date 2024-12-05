from sumo_rl import SumoEnvironment

class SUMOContextualEnv(SumoEnvironment):
    def change_context_reward_weight(self, pw, asw, dww, tqw):
        #change with context
        self.pressure_weight = 0
        self.avg_speed_weight = 0
        self.diff_wait_weight = 1
        self.total_queue_weight = -1

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

        self.traffic_signals[ts].last_reward = reward
        return reward
