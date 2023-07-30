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
 *  Tests for position embedding layer..
 */

import { DataType, Tensor, memory, ones, randomUniform, reshape, tidy, zeros } from '@tensorflow/tfjs-core';

import { Shape } from '../../../keras_format/common';
import { SymbolicTensor } from '../../../engine/topology';
import { Initializer } from '../../../initializers';
import { expectTensorsClose } from '../../../utils/test_utils';
import { input, model } from '../../../exports';
import { PositionEmbedding } from './position_embedding';

export class CustomInit extends Initializer {
  apply(shape: Shape, dtype?: DataType): Tensor {
    return tidy(() => {
      const count = shape.reduce((a, b) => a * b, 1);
      return reshape(Array(count).fill(1), shape).asType(dtype);
    });
  }
}

describe('PositionEmbedding Layer', () => {
  it('static layer output shape', () => {
    // Create a 3-dimensional input (the first dimension is implicit).
    const sequenceLength = 21;
    const featureSize = 30;
    const testLayer = new PositionEmbedding({sequenceLength});
    const inputTensor = input({shape: [sequenceLength, featureSize]});
    const outputTensor = testLayer.apply(inputTensor) as SymbolicTensor;

    // When using static position embedding shapes, the output is expected to
    // be the same shape as the input shape in all dimensions save batch.
    const expectedOutputShape = [null, sequenceLength, featureSize];

    expect(outputTensor.shape).toEqual(expectedOutputShape);
    // The output dtype for this layer should match the compute dtype.
    expect(outputTensor.dtype).toEqual(testLayer.dtype);
  });

  it('more than 3 dimensions static', () => {
    // Create a 4-dimensional input (the first dimension is implicit).
    const sequenceLength = 21;
    const featureSize = 30;
    const testLayer = new PositionEmbedding({sequenceLength});
    const inputTensor =
      input({shape: [featureSize, sequenceLength, featureSize]});
    const outputTensor = testLayer.apply(inputTensor) as SymbolicTensor;

    // When using static position embedding shapes, the output is expected
    // to be the same as the input shape in all dimensions save batch.
    const expectedOutputShape =
      [null, featureSize, sequenceLength, featureSize];

    expect(outputTensor.shape).toEqual(expectedOutputShape);
    // The output dtype for this layer should match the compute dtype.
    expect(outputTensor.dtype).toEqual(testLayer.dtype);
  });

  it('float32 dtype', () => {
    // Create a 3-dimensional input (the first dimension is implicit).
    const sequenceLength = 21;
    const featureSize = 30;
    const testLayer = new PositionEmbedding({sequenceLength, dtype: 'float32'});
    const inputTensor = input({shape: [sequenceLength, featureSize]});
    const outputTensor = testLayer.apply(inputTensor) as SymbolicTensor;

    // When using static position embedding shapes, the output is expected
    // to be the same as the input shape in all dimensions save batch.
    const expectedOutputShape =
      [null, sequenceLength, featureSize];

    expect(outputTensor.shape).toEqual(expectedOutputShape);
    // The output dtype for this layer should match the compute dtype.
    expect(outputTensor.dtype).toEqual('float32');
  });

  it('dynamic layer output shape', () => {
    const maxSequenceLength = 21;
    const featureSize = 30;
    const testLayer =
      new PositionEmbedding({sequenceLength: maxSequenceLength});
    // Create a 3-dimensional input (the first dimension is implicit).
    const inputTensor = input({shape: [null, featureSize]});
    const outputTensor = testLayer.apply(inputTensor) as SymbolicTensor;

    // When using dynamic position embedding shapes, the output is expected to
    // be the same shape as the input shape in all dimensions - but may be
    // null if the input shape is null there.
    const expectedOutputShape = [null, null, featureSize];

    expect(outputTensor.shape).toEqual(expectedOutputShape);
  });

  it('more than 3 dimensions dynamic', () => {
    const maxSequenceLength = 60;
    const featureSize = 30;
    const testLayer =
      new PositionEmbedding({sequenceLength: maxSequenceLength});
    // Create a 4-dimensional input (the first dimension is implicit).
    const inputTensor = input({shape: [null, null, featureSize]});
    const outputTensor = testLayer.apply(inputTensor) as SymbolicTensor;

    // When using dynamic position embedding shapes, the output is expected
    // to be the same as the input shape in all dimensions save batch.
    const expectedOutputShape = [null, null, null, featureSize];

    expect(outputTensor.shape).toEqual(expectedOutputShape);
  });

  it('dynamic layer slicing', () => {
    const maxSequenceLength = 40;
    const featureSize = 30;
    const testLayer =
      new PositionEmbedding({sequenceLength: maxSequenceLength});
    // Create a 3-dimensional input (the first dimension is implicit).
    const inputTensor = input({shape: [null, featureSize]});
    const outputTensor = testLayer.apply(inputTensor) as SymbolicTensor;

    const pmodel = model({inputs: inputTensor, outputs: outputTensor});

    // Create input data that is shorter than maxSequenceLength, which
    // should trigger a down-slice.
    const inputLength = 17;
    // Note: In practice, this layer should be used inside a model, where it can
    // be projected when added to another tensor.
    const inputData = ones([1, inputLength, featureSize]);
    const outputData = pmodel.predict(inputData) as Tensor;

    expect(outputData.shape).toEqual([1, inputLength, featureSize]);
  });

  it('callable initializer', () => {
    const maxSequenceLength = 4;
    const featureSize = 3;
    const testLayer = new PositionEmbedding({
      sequenceLength: maxSequenceLength,
      initializer: new CustomInit(),
    });
    const inputs = input({shape: [maxSequenceLength, featureSize]});
    const outputs = testLayer.apply(inputs) as SymbolicTensor;
    const pmodel = model({inputs, outputs});

    const batchSize = 2;
    const data = zeros([batchSize, maxSequenceLength, featureSize]);
    pmodel.apply(data);
    const modelOutput = pmodel.predict(data) as Tensor;
    const expectedOutput = reshape(
      Array.from({length: maxSequenceLength * featureSize}, (_, i) => i),
      [maxSequenceLength, featureSize],
    ).broadcastTo([batchSize, maxSequenceLength, featureSize]);

    expectTensorsClose(modelOutput, expectedOutput);
  });

  it('one training step', async () => {
    const maxSequenceLength = 4;
    const featureSize = 3;
    const inputs = input({shape: [maxSequenceLength, featureSize]});
    const testLayer =
      new PositionEmbedding({sequenceLength: maxSequenceLength});
    const outputs = testLayer.apply(inputs) as SymbolicTensor;
    const pmodel = model({inputs, outputs});

    const batchSize = 2;
    const data = randomUniform([batchSize, maxSequenceLength, featureSize]);
    const label = randomUniform([batchSize, maxSequenceLength, featureSize]);

    pmodel.compile({optimizer: 'adam', loss: 'meanSquaredError'});
    const loss = pmodel.trainOnBatch(data, label);

    expect(await loss).toBeGreaterThan(0);
  })

  it('serialization round trip', () => {
    const maxSequenceLength = 40;
    const testLayer = new PositionEmbedding({
      sequenceLength: maxSequenceLength,
      initializer: 'zeros',
    });
    const config = testLayer.getConfig();
    const restored = PositionEmbedding.fromConfig(PositionEmbedding, config);

    expect(restored.getConfig()).toEqual(config);
  });

  it('does not leak memory', () => {
    const sequenceLength = 4;
    const batchSize = 2;
    const testLayer = new PositionEmbedding({sequenceLength});
    const data = randomUniform([batchSize, sequenceLength, 3]);

    const numTensors = memory().numTensors;
    testLayer.call(data);

    expect(memory().numTensors).toEqual(numTensors + 1);
  });
});
