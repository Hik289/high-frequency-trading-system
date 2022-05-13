from numba import (
    int32, boolean, float64,
    typed, types,
)

# jitclass内变量的预设类型
kv1 = (types.string, float64[:, :])
kv2 = (types.string, boolean)
kv3 = (types.string, int32)
kv4 = (types.string, float64)
kv5 = (types.string, types.string)
kv6 = (types.string, int32[:, :])
kv7 = (int32, types.string)

handler_spec = [
    ('symbol_map', types.DictType(*kv7)),
    ('other_info', float64[:, :]),
    ('base_stat_info', types.ListType(types.string)),
    ('buffer_size_tick', int32),
    ('buffer_size_bar', int32),
    ('base_info_tick', types.ListType(types.string)),
    ('tick_buffer', float64[:, :, :]),
    ('tick_index', int32[:]),
    ('tick', float64[:]),
    ('base_info_bar', types.ListType(types.string)),
    ('bar_buffer', float64[:, :, :]),
    ('bar_index', int32),
    ('factor_tick_buffer', float64[:, :, :]),
    ('base_factor_info', types.ListType(types.string)),
    ('factor_bar_buffer', float64[:, :, :]),
    ('last_bar_use_tick_index', int32[:]),
    ('normal_tick_index', int32[:, :],),
    ('invoke_new_bar_seconds', int32),
    ('index_weight', float64[:, :]),
    ('days_of_last', int32),
    ('index_factor_buffer', float64[:, :]),
    ('index_bar_buffer', float64[:, :]),
    ('daily_factor', float64[:, :]),
    ('last_bar', float64[:, :, :]),
    ('last_factor', float64[:, :, :]),
    ('last_other_data', float64[:, :]),
    ('is_finished_last_bar', boolean),
    ('is_am', boolean),
    ('symbol_size', int32),
    ('factor_tick_size', int32),
    ('factor_bar_size', int32),
    ('bar_data', float64[:, :]),
    ('factor_data', float64[:, :]),
    ('ten_am_start_index', int32[:]),
    ('mid_day_stat', float64[:, :]),
    ('call_auction1', int32[:, :]),        # 9:15-9:20
    ('call_auction2', int32[:, :]),        # 9:20-9:25
    ('continuous_auction1', int32[:, :]),  # 9:30-10:00
    ('continuous_auction2', int32[:, :]),  # 10:00-14:45
    ('continuous_auction3', int32[:, :]),  # 14:45-14:57
    ('call_auction3', int32[:, :]),        # 14:57-15:30
    ('is_closed', int32[:]),
]


handler_kv0 = (types.string, float64[:, :])


# 默认TICK存储长度
BUFFER_SIZE_TICK = 6000
