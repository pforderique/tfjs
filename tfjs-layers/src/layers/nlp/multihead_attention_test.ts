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

import { Tensor, ones, randomUniform, randomUniformInt } from '@tensorflow/tfjs-core';

import { TruncatedNormal } from '../../initializers';
import { input } from '../../exports';
import { Shape } from '../../keras_format/common';
import { MultiHeadAttention } from './multihead_attention';
import { describeMathCPU, expectTensorsNotClose } from '../../utils/test_utils';

describe('MultiHeadAttention', () => {
  interface TestArgs {};

  interface NonMaskedAttentionArgs extends TestArgs {
    testcaseName: string;
    valueDim: number;
    outputShape: Shape;
    outputDims: Shape;
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

  // interface MaskedAttentionArgs extends TestArgs {
  //   testcaseName: string;
  //   useBias: boolean;
  // };
  // /**
  //  * Test with a mask tensor.
  //  */
  // function testMaskedAttention({testcaseName, useBias}: MaskedAttentionArgs) {
  //   it(`${testcaseName} masked attention`, () => {
  //     const testLayer = new MultiHeadAttention({
  //       numHeads: 2,
  //       keyDim: 2,
  //       useBias,
  //     });
  //     // Create a 3-dimensional input (the first dimension is implicit).
  //     const batchSize = 3;
  //     const query = input({shape: [4, 8]});
  //     const value = input({shape: [2, 8]});
  //     const attentionMask = input({shape: [4, 2]});
  //     const output =
  //       testLayer.apply(query, {value, attentionMask}) as SymbolicTensor;

  //     // Create a model containing the test layer.
  //     const mha =
  //       model({inputs: [query, value, attentionMask], outputs: output});

  //     function getRandomSample(shape: Shape, max: number): Tensor {
  //       const randomSample: number[][][] = [];
  //       for (let i = 0; i < shape[0]; i++) {
  //         const subArray: number[][] = [];
  //         for (let j = 0; j < shape[1]; j++) {
  //           const subSubArray: number[] = [];
  //           for (let k = 0; k < shape[2]; k++) {
  //             subSubArray.push(Math.random() * max);
  //           }
  //           subArray.push(subSubArray);
  //         }
  //         randomSample.push(subArray);
  //       }
  //       return tensor(randomSample);
  //     }

  //     function getRandomInt(max: number): number {
  //       return Math.floor(Math.random() * Math.floor(max));
  //     }

  //     function getRandomMaskData(shape: Shape): Tensor {
  //       const maskData: number[][][] = [];
  //       for (let i = 0; i < shape[0]; i++) {
  //         const subArray: number[][] = [];
  //         for (let j = 0; j < shape[1]; j++) {
  //           const subSubArray: number[] = [];
  //           for (let k = 0; k < shape[2]; k++) {
  //             subSubArray.push(getRandomInt(2));
  //           }
  //           subArray.push(subSubArray);
  //         }
  //         maskData.push(subArray);
  //       }
  //       return tensor(maskData);
  //     }

  //     // Generate data for the input (non-mask) tensors.
  //     const fromData = getRandomSample([batchSize, 4, 8], 10);
  //     const toData = getRandomSample([batchSize, 2, 8], 10);

  //     // Invoke the data with a random set of mask data. This should mask at
  //     // least one element.
  //     const maskData = getRandomMaskData([batchSize, 4, 2]);
  //     const maskedOutputData =
  //       mha.predict([fromData, toData, maskData]) as Tensor;

  //     // Invoke the same data, but with a null mask (where no elements are
  //     // masked).
  //     const nullMaskData = ones([batchSize, 4, 2]);
  //     const unmaskedOutputData =
  //       mha.predict([fromData, toData, nullMaskData]) as Tensor;

  //     expect(maskedOutputData.dataSync()).not.toEqual(
  //       unmaskedOutputData.dataSync());
  //   });
  // }
  // params = [
  //   {
  //     testcaseName: 'with bias',
  //     useBias: true,
  //   },
  //   {
  //     testcaseName: 'no bias',
  //     useBias: false,
  //   }
  // ];
  // for (const param of params) {
  //   testMaskedAttention(param as MaskedAttentionArgs);
  // }

  // Test with a specified initializer
  it('initializers', () => {
    const testLayer = new MultiHeadAttention({
      numHeads: 12,
      keyDim: 64,
      kernelInitializer: new TruncatedNormal({stddev: 0.02}),
    });
    const query = ones([1, 40, 80]);
    // TODO(pforderique): Once generic i/o is supported, change to call apply().
    const output = testLayer.call(query, {value: query}) as Tensor;
    expect(output.shape).toEqual([1, 40, 80]);

    // Make sure the sub layers have different kernel init value, and not
    // reusing the initializers.
    // TODO(pforderique): Debug why these kernels are the same. getInitializer
    // is returning a new instance - not the same one...
    // const queryKernel = testLayer.queryDense.kernel.read();
    // const keyKernel = testLayer.keyDense.kernel.read();
    // const valueKernel = testLayer.valueDense.kernel.read();
    // const outputKernel = testLayer.outputDense.kernel.read();

    // expectTensorsNotClose(queryKernel, keyKernel);
    // expectTensorsNotClose(queryKernel, valueKernel);
    // expectTensorsNotClose(queryKernel, outputKernel);
  });

  describeMathCPU('High dimensions', () => {
    interface HighDimAttentionArgs extends TestArgs {
      testcaseName: string;
      qDims: Shape;
      vDims: Shape;
      maskDims: Shape;
      attentionAxes: number[];
    };
    /**
     * Test with high dimensional inputs.
     */
    function testHighDimAttention({
      testcaseName, qDims, vDims, maskDims, attentionAxes,
    }: HighDimAttentionArgs) {
      it(`${testcaseName} high dim attention`, () => {
        const testLayer = new MultiHeadAttention({
          numHeads: 2, keyDim: 2, attentionAxes,
        });
        const batchSize = 3;
        const hiddenSize = 8;
        // Generate data for the input (non-mask) tensors.
        const queryShape = [batchSize].concat(qDims).concat(hiddenSize);
        const valueShape = [batchSize].concat(vDims).concat(hiddenSize);
        const maskShape = [batchSize].concat(maskDims);
        const query = randomUniform(queryShape, 0, 10);
        const value = randomUniform(valueShape, 0, 10);

        // Invoke the data with a random set of mask data. This should mask at
        // least one element.
        const maskData = randomUniformInt(maskShape, 0, 2).asType('bool');

        // Invoke the same data, but with a null mask (where no elements are
        // masked).
        const nullMaskData = ones(maskShape);

        // Because one data is masked and one is not, the outputs should not be
        // the same.

        const outputWithMask = testLayer.call(
          query, {value, attentionMask: maskData});
        const outputWithNullMask = testLayer.call(
          query, {value, attentionMask: nullMaskData});

        expectTensorsNotClose(outputWithMask, outputWithNullMask);
      });
    }
    params = [
      {
        testcaseName: '4d_inputs_1freebatch_mask2',
        qDims: [3, 4],
        vDims: [3, 2],
        maskDims: [4, 2],
        attentionAxes: [2],
      },
      {
        testcaseName: '4d_inputs_1freebatch_mask3',
        qDims: [3, 4],
        vDims: [3, 2],
        maskDims: [3, 4, 2],
        attentionAxes: [2],
      },
      {
        testcaseName: '4d_inputs_1freebatch_mask4',
        qDims: [3, 4],
        vDims: [3, 2],
        maskDims: [3, 2, 4, 2],
        attentionAxes: [2],
      },
      // TODO(pforderique): Add test cases '4D_inputs_2D_attention',
      // '5D_inputs_2D_attention', and '5D_inputs_2D_attention_fullmask' once
      // GPU for rank 7 tensors is supported.
      {
        testcaseName: '4D_inputs_2D_attention',
        qDims: [3, 4],
        vDims: [3, 2],
        maskDims: [3, 4, 3, 2],
        attentionAxes: [1, 2],
      },
      {
        testcaseName: '5D_inputs_2D_attention',
        qDims: [5, 3, 4],
        vDims: [5, 3, 2],
        maskDims: [3, 4, 3, 2],
        attentionAxes: [2, 3],
      },
      {
        testcaseName: '5D_inputs_2D_attention_fullmask',
        qDims: [5, 3, 4],
        vDims: [5, 3, 2],
        maskDims: [5, 3, 4, 3, 2],
        attentionAxes: [2, 3],
      },
    ];
    for (const param of params) {
      // testHighDimAttention; param;
      testHighDimAttention(param as HighDimAttentionArgs);
    }
  });
  // TODO(pforderique): Test memory and serialization.
});
