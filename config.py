from collections import OrderedDict

COLOR_ENCODING = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('water', (61, 230, 250)),
    ('building-no-damage', (180, 120, 120)),
    ('building-medium-damage', (235, 255, 7)),
    ('building-major-damage', (255, 184, 6)),
    ('building-total-destruction', (255, 0, 0)),
    ('vehicle', (255, 0, 245)),
    ('road-clear', (140, 140, 140)),
    ('road-blocked', (160, 150, 20)),
    ('tree', (4, 250, 7)),
    ('pool', (255, 235, 0)),
])

# Веса классов можно подстроить после анализа дисбаланса в датасете
CLASS_WEIGHTS = [
    0.2,  # unlabeled
    1.0,  # water
    1.5,  # building-no-damage
    2.0,  # building-medium-damage
    2.5,  # building-major-damage
    3.0,  # building-total-destruction
    2.0,  # vehicle
    1.0,  # road-clear
    2.0,  # road-blocked
    1.2,  # tree
    1.3   # pool
]