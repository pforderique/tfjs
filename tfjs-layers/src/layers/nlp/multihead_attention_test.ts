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

import { Tensor, ones } from '@tensorflow/tfjs-core';

import { input, model } from '../../exports';
import { Shape } from '../../keras_format/common';
import { MultiHeadAttention } from './multihead_attention';

describe('MultiHeadAttention', () => {

  interface NonMaskedAttentionArgs {
    testcaseName: string,
    valueDim: number,
    outputShape: Shape,
    outputDims: Shape
  }
  /**
   * Test that the attention layer can be created without a mask tensor.
   */
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

  // Test with one input (self-attenntion) and no mask tensor.
  it('non masked self attention', () => {
    const testLayer = new MultiHeadAttention({numHeads: 12, keyDim: 64});
    // Create a 3-dimensional input (the first dimension is implicit).
    const query = input({shape: [40, 80]});
    const output = testLayer.apply(query) as Tensor;
    expect(output.shape).toEqual([null, 40, 80]);
  });

  // Test attention outputs with coefficients.
  it('attention scores', () => {
    const testLayer = new MultiHeadAttention({numHeads: 12, keyDim: 64});
    // Create a 3-dimensional input (the first dimension is implicit).
    const query = ones([40, 80]);
    const [output, coef] =
      testLayer.callAndReturnAttentionScores(query, {value: query});
    expect(output.shape).toEqual([null, 40, 80]);
    expect(coef.shape).toEqual([null, 12, 40, 40]);
  });

  // Test attention outputs with coefficients.
  it('attention scores with values', () => {
    const testLayer = new MultiHeadAttention({numHeads: 12, keyDim: 64});
    // Create a 3-dimensional input (the first dimension is implicit).
    const query = ones([40, 80]);
    const value = ones([60, 80]);
    const [output, coef] =
      testLayer.callAndReturnAttentionScores(query, {value});
    expect(output.shape).toEqual([null, 40, 80]);
    expect(coef.shape).toEqual([null, 12, 40, 60]);
  });

  interface MaskedAttentionArgs {
    testcaseName: string,
    useBias: boolean,
  };
  /**
   * Test with a mask tensor.
   */
  function testMaskedAttention({testcaseName, useBias}: MaskedAttentionArgs) {
    it(`${testcaseName} masked attention`, () => {
      const testLayer = new MultiHeadAttention({
        numHeads: 2,
        keyDim: 2,
        useBias,
      });
      // Create a 3-dimensional input (the first dimension is implicit).
      const batchSize = 3;
      const query = input({shape: [4, 8]});
      const value = input({shape: [2, 8]});
      const attentionMask = input({shape: [4, 2]});
      const output = testLayer.apply(query, {value, attentionMask}) as Tensor;

      // Create a model containing the test layer.
      // ! Left off here. Perhaps do this test case later? const modes = model()
    });
  }
  // TODO(pforderique): Test memory and serialization.
});
