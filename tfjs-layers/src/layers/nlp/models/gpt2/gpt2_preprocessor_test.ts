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
 * Unit Tests for GPT2Preprocessor.
 */

import { tensor } from '@tensorflow/tfjs-core';

import { expectTensorsClose } from '../../../../utils/test_utils';
import { GPT2Tokenizer } from './gpt2_tokenizer';
import { GPT2Preprocessor, PreprocessorOutputs } from './gpt2_preprocessor';
import { tensorArrTo2DArr } from '../../tokenizers_utils';

describe('GPT2Preprocessor', () => {
  let vocabulary: Map<string, number>;
  let merges: string[];
  let preprocessor: GPT2Preprocessor;

  beforeEach(() => {
    vocabulary = new Map([
      ['!', 0],
      ['air', 1],
      ['Ġair', 2],
      ['plane', 3],
      ['Ġat', 4],
      ['port', 5],
      ['<|endoftext|>', 6],
    ]);

    merges = ['Ġ a', 'Ġ t', 'Ġ i', 'Ġ b', 'a i', 'p l', 'n e'].concat(
      ['Ġa t', 'p o', 'r t', 'Ġt h', 'ai r', 'pl a', 'po rt'],
      ['Ġai r', 'Ġa i', 'pla ne']
    );
    preprocessor = new GPT2Preprocessor({
      tokenizer: new GPT2Tokenizer({vocabulary, merges}),
      sequenceLength: 8.
    });
  });

  it('tokenize', () => {
    const inputData = tensor(['airplane at airport']);

    const output =
      preprocessor.callAndPackArgs(inputData, {}) as PreprocessorOutputs;

    expectTensorsClose(output.tokenIds[0], tensor([6, 1, 3, 4, 2, 5, 6, 0]));
    expectTensorsClose(
      output.paddingMask[0], tensor([1, 1, 1, 1, 1, 1, 1, 0], [8], 'bool'));
  });

  it('no start end token', () => {
    const inputData = tensor(Array<string>(2).fill('airplane at airport'));
    preprocessor = new GPT2Preprocessor({
      tokenizer: new GPT2Tokenizer({vocabulary, merges}),
      sequenceLength: 8,
      addStartToken: false,
      addEndToken: false,
    });
    const expectedOutput = {
      tokenIds: Array<number[]>(2).fill([1, 3, 4, 2, 5, 0, 0, 0]),
      paddingMask: Array<number[]>(2).fill([1, 1, 1, 1, 1, 0, 0, 0]),
    }

    const output =
      preprocessor.callAndPackArgs(inputData, {}) as PreprocessorOutputs;

    const outputTokenIds = tensorArrTo2DArr(output.tokenIds) as number[][];
    const outputMask = tensorArrTo2DArr(output.paddingMask) as number[][];

    // console.log('outputTokenIds', outputTokenIds);
    // console.log('outputTokenIds non number',  Array.from(outputTokenIds));

    expect(outputTokenIds).toEqual(expectedOutput.tokenIds);
    expect(outputMask).toEqual(expectedOutput.paddingMask);
  });
});
