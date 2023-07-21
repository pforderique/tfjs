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

import { Tensor, ones, tensor } from '@tensorflow/tfjs-core';

import { Shape } from '../../keras_format/common';
import { MultiHeadAttention } from './multihead_attention';
import { SymbolicTensor } from '../../engine/topology';
import { input, model } from '../../exports';

describe('MultiHeadAttention', () => {
  interface TestArgs {};

  interface NonMaskedAttentionArgs extends TestArgs {
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

  let params: TestArgs[] = [
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
    testNonMaskedAttention(param as NonMaskedAttentionArgs);
  }

  // Test with one input (self-attenntion) and no mask tensor.
  it('non masked self attention', () => {
    const testLayer = new MultiHeadAttention({numHeads: 12, keyDim: 64});
    // Create a 3-dimensional input (the first dimension is implicit).
    const query = input({shape: [40, 80]});
    const output = testLayer.apply(query, {value: query}) as Tensor;
    expect(output.shape).toEqual([null, 40, 80]);
  });

  // Test attention outputs with coefficients.
  it('attention scores', () => {
    const testLayer = new MultiHeadAttention({numHeads: 12, keyDim: 64});
    const query = ones([1, 40, 80]);
    const [output, coef] =
      testLayer.callAndReturnAttentionScores(query, {value: query});
    expect(output.shape).toEqual([1, 40, 80]);
    expect(coef.shape).toEqual([1, 12, 40, 40]);
  });

  // Test attention outputs with coefficients.
  it('attention scores with values', () => {
    const testLayer = new MultiHeadAttention({numHeads: 12, keyDim: 64});
    const query = ones([1, 40, 80]);
    const value = ones([1, 60, 80]);
    const [output, coef] =
      testLayer.callAndReturnAttentionScores(query, {value});
    expect(output.shape).toEqual([1, 40, 80]);
    expect(coef.shape).toEqual([1, 12, 40, 60]);
  });

  interface MaskedAttentionArgs extends TestArgs {
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
      const output =
        testLayer.apply(query, {value, attentionMask}) as SymbolicTensor;

      // Create a model containing the test layer.
      const mha =
        model({inputs: [query, value, attentionMask], outputs: output});

      function getRandomSample(shape: Shape, max: number): Tensor {
        const randomSample: number[][][] = [];
        for (let i = 0; i < shape[0]; i++) {
          const subArray: number[][] = [];
          for (let j = 0; j < shape[1]; j++) {
            const subSubArray: number[] = [];
            for (let k = 0; k < shape[2]; k++) {
              subSubArray.push(Math.random() * max);
            }
            subArray.push(subSubArray);
          }
          randomSample.push(subArray);
        }
        return tensor(randomSample);
      }

      function getRandomInt(max: number): number {
        return Math.floor(Math.random() * Math.floor(max));
      }

      function getRandomMaskData(shape: Shape): Tensor {
        const maskData: number[][][] = [];
        for (let i = 0; i < shape[0]; i++) {
          const subArray: number[][] = [];
          for (let j = 0; j < shape[1]; j++) {
            const subSubArray: number[] = [];
            for (let k = 0; k < shape[2]; k++) {
              subSubArray.push(getRandomInt(2));
            }
            subArray.push(subSubArray);
          }
          maskData.push(subArray);
        }
        return tensor(maskData);
      }

      // Generate data for the input (non-mask) tensors.
      const fromData = getRandomSample([batchSize, 4, 8], 10);
      const toData = getRandomSample([batchSize, 2, 8], 10);

      // Invoke the data with a random set of mask data. This should mask at
      // least one element.
      const maskData = getRandomMaskData([batchSize, 4, 2]);
      const maskedOutputData =
        mha.predict([fromData, toData, maskData]) as Tensor;

      // Invoke the same data, but with a null mask (where no elements are
      // masked).
      const nullMaskData = ones([batchSize, 4, 2]);
      const unmaskedOutputData =
        mha.predict([fromData, toData, nullMaskData]) as Tensor;

      expect(maskedOutputData.dataSync()).not.toEqual(
        unmaskedOutputData.dataSync());
    });
  }
  params = [
    {
      testcaseName: 'with bias',
      useBias: true,
    },
    {
      testcaseName: 'no bias',
      useBias: false,
    }
  ];
  for (const param of params) {
    testMaskedAttention(param as MaskedAttentionArgs);
  }
  // TODO(pforderique): Test memory and serialization.
});
