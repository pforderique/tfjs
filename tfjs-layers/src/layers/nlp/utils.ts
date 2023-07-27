/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import { Tensor, tensorScatterUpdate } from '@tensorflow/tfjs-core';

export function tensorToArr(input: Tensor): unknown[] {
  return Array.from(input.dataSync()) as unknown as unknown[];
}

export function tensorArrTo2DArr(inputs: Tensor[]): unknown[][] {
  return inputs.map(input => tensorToArr(input));
}

export function sliceUpdate(
    inputs: Tensor, startIndices: number[], updates: Tensor): Tensor {
  const indices: number[][] = [];
  function createIndices(idx: number, curr: number[]): void {
    if (curr.length === startIndices.length) {
      indices.push(curr.slice());
      return;
    }
    const s = startIndices[idx];
    const e = s + updates.shape[idx];
    for (let i = s; i < e; i++) {
      curr.push(i);
      createIndices(idx + 1, curr);
      curr.pop();
    }
  }
  createIndices(0, []);
  // Flatten the updates to match its update indices.
  updates = updates.reshape([updates.size]);
  return tensorScatterUpdate(inputs, indices, updates);
}
