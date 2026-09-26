"""Independent conservation checks from model stream tables (kcal/h, kg/h)."""


def stream_balances(result):
    streams = {stream['no']: stream for stream in result['streams']}
    units = {
        'EV101': ([13, 28], [14, 26, 29]),
        'EV201': ([7, 26], [8, 19, 24]),
        'EV301': ([1, 21], [2, 18, 23]),
        'E101': ([9, 15], [11, 16]),
        'E102': ([10, 29], [12, 30]),
        'E201': ([3, 16], [5, 17]),
        'E202': ([4, 30], [6, 31]),
        'flash': ([24], [20, 25]),
        'condenser': ([18], [22]),
        'condensate_mix': ([22, 23, 25], [27]),
        'global': ([1, 28], [17, 27, 31]),
    }
    results = {}
    for name, (incoming, outgoing) in units.items():
        mass = sum(streams[i]['F_kg_h'] for i in incoming) - sum(streams[i]['F_kg_h'] for i in outgoing)
        energy = sum(streams[i]['F_kg_h'] * streams[i]['h_kcal_kg'] for i in incoming)
        energy -= sum(streams[i]['F_kg_h'] * streams[i]['h_kcal_kg'] for i in outgoing)
        if name in ('condenser', 'global'):
            energy -= result['duty_mcal_h']['COND'] * 1e6
        results[name] = {'mass_kg_h': float(mass), 'energy_kcal_h': float(energy)}
    return results
