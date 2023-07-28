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
import { memory, ones, randomUniform, zeros } from '@tensorflow/tfjs-core';

import { CachedMultiHeadAttention } from './cached_multihead_attention';
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
    decoder.apply(decoderInput, {encoderSequence: encoderInput})

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

  it('does not leak memory', () => {
    const layer = new CachedMultiHeadAttention({numHeads: 2, keyDim: 2});
    const query = ones([1, 4, 8]);
    // Initial call that builds sublayers and necessary tensors.
    layer.call(query, {value: query});

    const numTensors = memory().numTensors;
    layer.call(query, {value: query});

    expect(memory().numTensors).toEqual(numTensors + 1);
  });
  // TODO(pforderique): Test serialization.
});
