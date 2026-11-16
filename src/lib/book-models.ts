/** Educational models, not measurements of a particular GPU or model. */
export function weightMemoryGB(parametersBillions: number, bits: number): number {
  return parametersBillions * bits / 8;
}

export function rooflinePoint(intensity: number, bandwidthGBs: number, peakTFLOPs: number) {
  const memoryTFLOPs = intensity * bandwidthGBs / 1000;
  return {
    attainable: Math.min(memoryTFLOPs, peakTFLOPs),
    ridge: peakTFLOPs * 1000 / bandwidthGBs,
    bound: memoryTFLOPs < peakTFLOPs ? 'Memory-bound' : 'Compute-bound',
  };
}

/** Full causal attention: token positions whose K/V projections are computed. */
export function kvProjectionWork(promptTokens: number, newTokens: number) {
  return {
    uncached: Array.from({length: newTokens}, (_, i) => promptTokens + i + 1)
      .reduce((sum, n) => sum + n, 0),
    cached: newTokens,
  };
}

/** Unit-length decode steps, all requests available at t=0, fixed slots. */
export function batchSchedule(lengths: number[], slots: number, continuous: boolean) {
  type Entry = {request: number; step: number};
  const rows: (Entry | null)[][] = [];
  if (!continuous) {
    for (let start = 0; start < lengths.length; start += slots) {
      const group = lengths.slice(start, start + slots);
      for (let step = 0; step < Math.max(...group); step++) {
        rows.push(Array.from({length: slots}, (_, i) =>
          i < group.length && step < group[i] ? {request: start + i, step} : null));
      }
    }
    return rows;
  }
  type Active = {length: number; request: number; step: number};
  const active: (Active | null)[] = Array.from({length: slots}, (_, request) =>
    request < lengths.length ? {length: lengths[request], request, step: 0} : null);
  let next = slots;
  while (active.some(Boolean)) {
    rows.push(active.map(entry => entry ? {request: entry.request, step: entry.step} : null));
    for (let i = 0; i < active.length; i++) {
      const entry = active[i];
      if (!entry) continue;
      entry.step++;
      if (entry.step >= entry.length) {
        active[i] = next < lengths.length ? {length: lengths[next], request: next++, step: 0} : null;
      }
    }
  }
  return rows;
}
