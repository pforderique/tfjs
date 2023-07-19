/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Unit Tests for MultiHeadAttention layer.
 */

import { Tensor } from '@tensorflow/tfjs-core';

import { input } from '../../exports';
import { Shape } from '../../keras_format/common';
import { MultiHeadAttention } from './multihead_attention';

describe('MultiHeadAttention', () => {

  /**
   * Test that the attention layer can be created without a mask tensor.
   */
  interface NonMaskedAttentionArgs {
    testcaseName: string,
    valueDim: number,
    outputShape: Shape,
    outputDims: Shape
  }
  function testNonMaskedAttention(
    {testcaseName, valueDim, outputShape, outputDims}: NonMaskedAttentionArgs
  ) {
    it(`${testcaseName} non masked attention`, () => {
      const testLayer = new MultiHeadAttention({
        numHeads: 12,
        keyDim: 64,
        valueDim,
        outputShape,
      });
      // Create a 3-dimensional input (the first dimension is implicit).
      const query = input({shape: [40, 80]});
      const value = input({shape: [20, 80]});
      const output = testLayer.apply(query, {value}) as Tensor;
      expect(output.shape).toEqual([null].concat(outputDims));
    });
  }

  let params: NonMaskedAttentionArgs[] = [
    {
      testcaseName: 'key value same proj',
      valueDim: null,
      outputShape: null,
      outputDims: [40, 80],
    },
    {
      testcaseName: 'key value different proj',
      valueDim: 32,
      outputShape: [60],
      outputDims: [40, 60],
    }
  ];
  for (const param of params) {
    testNonMaskedAttention(param);
  }

  // TODO(pforderique): Test memory and serialization.
});
