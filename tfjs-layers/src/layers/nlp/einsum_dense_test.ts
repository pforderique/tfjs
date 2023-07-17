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

/**
 * Unit Tests for TFJS-based EinsumDense Layer.
 */

import { Tensor } from '@tensorflow/tfjs-core';

import { Shape } from '../../keras_format/common';
import { analyzeEinsumString, EinsumDense } from './einsum_dense';
import { input } from '../../exports';

declare interface EinsumDenseTestCaseArgs {
  testcaseName: string;
  equation: string;
  biasAxes: string;
  inputShape: Shape;
  outputShape: Shape;
  expectedWeightShape: Shape;
  expectedBiasShape: Shape;
  expectedOutputShape: Shape;
}

describe('EinsumDense', () => {
  const combinations: EinsumDenseTestCaseArgs[] = [
    {
      testcaseName: '_1d_end_weight',
      equation: 'ab,b->a',
      biasAxes: null,
      inputShape: [null, 32],
      outputShape: [],
      expectedWeightShape: [32],
      expectedBiasShape: null,
      expectedOutputShape: [null],
    }
  ];
  // let einsumDense: EinsumDense;

  beforeEach(() => {
    // einsumDense = new EinsumDense({equation: 'ab,b->a', outputShape: [32]});
  });

  function testWeightShape(combo: EinsumDenseTestCaseArgs) {
    it(`${combo.testcaseName} weight shape`, () => {
      const [weightShape, biasShape, _] = analyzeEinsumString(
        combo.equation, combo.biasAxes, combo.inputShape, combo.outputShape
      );

      expect(weightShape).toEqual(combo.expectedWeightShape);
      expect(biasShape).toEqual(combo.expectedBiasShape);
    });
  }

  function testLayerCreation(combo: EinsumDenseTestCaseArgs) {
    it(`${combo.testcaseName} layer creation`, () => {
      const nonBatchInputShape = combo.inputShape.slice(1);
      const inputTensor = input({shape: nonBatchInputShape});

      const layer = new EinsumDense({
        equation: combo.equation,
        biasAxes: combo.biasAxes,
        outputShape: combo.outputShape,
      });
      const outputTensor = layer.apply(inputTensor) as Tensor;

      expect(layer.kernel.shape).toEqual(combo.expectedWeightShape);
      if (combo.expectedBiasShape != null) {
        expect(layer.bias).toBeNull();
      } else {
        expect(layer.bias.shape).toEqual(combo.expectedBiasShape);
      }
      expect(outputTensor.shape).toEqual(combo.expectedOutputShape);
    });
  }

  for (const combo of combinations) {
    testWeightShape(combo);
    testLayerCreation(combo);
  }
});
