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
 * Unit Tests for Transformer Decoder.
 */
import { Tensor, memory, randomUniform, randomUniformInt, tensor, zeros, zerosLike } from '@tensorflow/tfjs-core';

import { SymbolicTensor } from '../../../engine/topology';
import { input, model } from '../../../exports';
import { expectTensorsClose } from '../../../utils/test_utils';
import { Dense } from '../../core';
import { sliceUpdate } from '../utils';

import { TransformerDecoder } from './transformer_decoder';

describe('TransformerDecoder', () => {
  describe('valid call', () => {
    function testValidCall(testcaseName: string, normalizeFirst: boolean) {
      it(testcaseName, () => {
        const encoderInput = randomUniform([4, 6]);
        const decoderInput = randomUniform([4, 6]);
        const decoder = new TransformerDecoder({
          intermediateDim: 4,
          numHeads: 2,
          normalizeFirst,
        });
        expect(
          () => decoder.apply(decoderInput, {encoderSequence: encoderInput})
        ).not.toThrow();
      });
    }
    function testValidCallWithoutCrossAttention(
        testcaseName: string, normalizeFirst: boolean) {
      it(`${testcaseName} without cross attention`, () => {
        const encoderInput = randomUniform([4, 6]);
        const decoderInput = randomUniform([4, 6]);
        const decoder = new TransformerDecoder({
          intermediateDim: 4,
          numHeads: 2,
          normalizeFirst,
        });
        expect(
          () => decoder.apply(decoderInput, {encoderSequence: encoderInput})
        ).not.toThrow();
      });
    }

    const params: Array<[string, boolean]> = [
      ['without_norm_first', false],
      ['with_norm_first', true],
    ];

    for (const [testcaseName, normalizeFirst] of params) {
      testValidCall(testcaseName, normalizeFirst);
      testValidCallWithoutCrossAttention(testcaseName, normalizeFirst);
    }
  });

  it('invalid call', () => {
    const encoderInput = zeros([4, 6]);
    const decoderInput = zeros([4, 6]);
    // With cross-attention.
    let decoder = new TransformerDecoder({intermediateDim: 4, numHeads: 2});
    decoder.apply(decoderInput, {encoderSequence: encoderInput});

    // Should raise ValueError if encoderInput is not provided.
    expect(() => decoder.apply(decoderInput)).toThrow();

    // Without cross-attention.
    decoder = new TransformerDecoder({intermediateDim: 4, numHeads: 2});
    decoder.apply(decoderInput);

    // Should raise ValueError if encoderInput is provided.
    expect(
      () => decoder.apply(decoderInput, {encoderSequence: encoderInput})
    ).toThrow();
  });

  it('error when invalid kernerl initializer', () => {
    expect(() => new TransformerDecoder({
      intermediateDim: 4,
      numHeads: 2,
      dropout: 0.5,
      kernelInitializer: 'Invalid',
    })).toThrow();
  });

  it('one training step of transformer with cross attention', async () => {
    const decoderInput = input({shape: [4, 6]});
    const encoderInput = input({shape: [4, 6]});
    const decoder = new TransformerDecoder({intermediateDim: 4, numHeads: 2});
    let outputs = decoder.apply(decoderInput, {encoderSequence: encoderInput});
    outputs = new Dense({
      units: 10, activation: 'softmax'}).apply(outputs) as SymbolicTensor;
    const tModel = model({inputs: [decoderInput, encoderInput], outputs});

    const decoderSequence = randomUniform([2, 4, 6]);
    const encoderSequence = randomUniform([2, 4, 6]);
    const label = randomUniformInt([2, 4, 1], 0, 10);

    tModel.compile({loss: 'sparseCategoricalCrossentropy', optimizer: 'adam'});
    const loss = tModel.trainOnBatch([decoderSequence, encoderSequence], label);

    expect(await loss).toBeGreaterThan(0);
  });

  it('one training step of transformer without cross attention', async () => {
    const decoderInput = input({shape: [4, 6]});
    const decoder = new TransformerDecoder({intermediateDim: 4, numHeads: 2});
    let outputs = decoder.apply(decoderInput);
    outputs = new Dense({
      units: 10, activation: 'softmax'}).apply(outputs) as SymbolicTensor;
    const tModel = model({inputs: decoderInput, outputs});

    const decoderSequence = randomUniform([2, 4, 6]);
    const label = randomUniformInt([2, 4, 1], 0, 10);

    tModel.compile({loss: 'sparseCategoricalCrossentropy', optimizer: 'adam'});
    const loss = tModel.trainOnBatch(decoderSequence, label);

    expect(await loss).toBeGreaterThan(0);
  });

  it('mask propogation', () => {
    const decoder = new TransformerDecoder({intermediateDim: 4, numHeads: 2});
    const decoderSequence = randomUniform([1, 4, 6]);
    const encoderSequence = randomUniform([1, 4, 6]);
    const decoderAttentionMask = tensor([[1, 1, 0, 0]], [1, 4], 'bool');
    const outputs = decoder.apply(
      decoderSequence,
      {encoderSequence, decoderAttentionMask}
    ) as Tensor;

    expectTensorsClose(
      decoder.computeMask(outputs, decoderAttentionMask) as Tensor,
      decoderAttentionMask
    );
  });

  it('mask propogation without cross attention ', () => {
    const decoder = new TransformerDecoder({intermediateDim: 4, numHeads: 2});
    const decoderSequence = randomUniform([1, 4, 6]);
    const decoderAttentionMask = tensor([[1, 1, 0, 0]], [1, 4], 'bool');
    const outputs = decoder.apply(
      decoderSequence,
      {decoderAttentionMask}
    ) as Tensor;

    expectTensorsClose(
      decoder.computeMask(outputs, decoderAttentionMask) as Tensor,
      decoderAttentionMask
    );
  });

  it('cache call is correct', () => {
    const batchSize = 2;
    const seqLen = 5;
    const numHeads = 2;
    const keyDim = 4;
    const hiddenDim = numHeads * keyDim;

    const inputShape = [batchSize, seqLen, hiddenDim];
    const x = randomUniform(inputShape);
    const inputCache = zeros([batchSize, 2, seqLen, numHeads, keyDim]);
    const outputs = zerosLike(x);

    const layer = new TransformerDecoder({intermediateDim: 4, numHeads});
    const [noLoopOutputs, noLoopCache] = layer.callAndReturnCaches(
      x, {selfAttentionCache: inputCache, selfAttentionCacheUpdateIndex: 0});

    function call(outputs: Tensor, cache: Tensor) {
      for (let i = 0; i < seqLen; i++) {
        // Compute the rest tokens.
        const nextInput = x.slice([0, i, 0], [batchSize, 1, hiddenDim]);
        const [nextOutput, nextCache] = layer.callAndReturnCaches(
          nextInput,
          {
            selfAttentionCache: cache,
            selfAttentionCacheUpdateIndex: i
          }
        );
        outputs = sliceUpdate(outputs, [0, i, 0], nextOutput);
        cache = nextCache;
      }
      return [outputs, cache];
    }
    const [output, outputCache] = call(outputs, inputCache);

    expectTensorsClose(output, noLoopOutputs);
    expectTensorsClose(outputCache, noLoopCache);
  });

  it('serialization round trip', () => {
    const testLayer = new TransformerDecoder({intermediateDim: 4, numHeads: 2});

    const config = testLayer.getConfig();
    const restored = TransformerDecoder.fromConfig(TransformerDecoder, config);

    expect(restored.getConfig()).toEqual(config);
  });

  it('does not leak memory', () => {
    const encoderInput = randomUniform([4, 6]);
    const decoderInput = randomUniform([4, 6]);
    const decoder = new TransformerDecoder({intermediateDim: 4, numHeads: 2});

    const numTensors = memory().numTensors;
    decoder.call(decoderInput, {encoderSequence: encoderInput});

    expect(memory().numTensors).toEqual(numTensors + 1);
  });
});
