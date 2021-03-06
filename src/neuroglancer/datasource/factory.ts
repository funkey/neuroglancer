/**
 * @license
 * Copyright 2016 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {ChunkManager} from 'neuroglancer/chunk_manager/frontend';
import {MeshSource} from 'neuroglancer/mesh/frontend';
import {SkeletonSource} from 'neuroglancer/skeleton/frontend';
import {MultiscaleVolumeChunkSource} from 'neuroglancer/sliceview/frontend';
import {CompletionWithDescription, applyCompletionOffset} from 'neuroglancer/util/completion';
import {CancellablePromise, cancellableThen} from 'neuroglancer/util/promise';

export type Completion = CompletionWithDescription;

export interface CompletionResult {
  offset: number;
  completions: Completion[];
}

/**
 * Returns the length of the prefix of path that corresponds to the "group", according to the
 * specified separator.
 *
 * If the separator is not specified, gueses whether it is '/' or ':'.
 */
export function findSourceGroupBasedOnSeparator(path: string, separator?: string) {
  if (separator === undefined) {
    // Try to guess whether '/' or ':' is the separator.
    if (path.indexOf('/') === -1) {
      separator = ':';
    } else {
      separator = '/';
    }
  }
  let index = path.lastIndexOf(separator);
  if (index === -1) {
    return 0;
  }
  return index + 1;
}


/**
 * Returns the last "component" of path, according to the specified separator.
 * If the separator is not specified, gueses whether it is '/' or ':'.
 */
export function suggestLayerNameBasedOnSeparator(path: string, separator?: string) {
  let groupIndex = findSourceGroupBasedOnSeparator(path, separator);
  return path.substring(groupIndex);
}

export interface DataSourceFactory {
  description?: string;
  getVolume?: (path: string) => Promise<MultiscaleVolumeChunkSource>;
  getMeshSource?: (chunkManager: ChunkManager, path: string, lod: number) => MeshSource;
  getSkeletonSource?: (chunkManager: ChunkManager, path: string) => SkeletonSource;
  volumeCompleter?: (value: string) => CancellablePromise<CompletionResult>;

  /**
   * Returns a suggested layer name for the given volume source.
   */
  suggestLayerName?: (path: string) => string;

  /**
   * Returns the length of the prefix of path that is its 'group'.  This is used for suggesting a
   * default URL for adding a new layer.
   */
  findSourceGroup?: (path: string) => number;
}

const dataSourceFactories = new Map<string, DataSourceFactory>();

export function registerDataSourceFactory(name: string, factory: DataSourceFactory) {
  dataSourceFactories.set(name, factory);
}

const protocolPattern = /^(?:([a-zA-Z-+_]+):\/\/)?(.*)$/;

function getDataSource(url: string): [DataSourceFactory, string, string] {
  let m = url.match(protocolPattern);
  if (m === null || m[1] === undefined) {
    throw new Error(`Data source URL must have the form "<protocol>://<path>".`);
  }
  let dataSource = m[1];
  let factory = dataSourceFactories.get(dataSource);
  if (factory === undefined) {
    throw new Error(`Unsupported data source: ${JSON.stringify(dataSource)}.`);
  }
  return [factory, m[2], dataSource];
}

export function getVolume(url: string) {
  let [factories, path] = getDataSource(url);
  return factories.getVolume!(path);
}

export function getMeshSource(chunkManager: ChunkManager, url: string, lod: number = 0) {
  let [factories, path] = getDataSource(url);
  return factories.getMeshSource!(chunkManager, path, lod);
}

export function getSkeletonSource(chunkManager: ChunkManager, url: string) {
  let [factories, path] = getDataSource(url);
  return factories.getSkeletonSource!(chunkManager, path);
}

export function volumeCompleter(url: string): CancellablePromise<CompletionResult> {
  // Check if url matches a protocol.  Note that protocolPattern always matches.
  let protocolMatch = url.match(protocolPattern)!;
  let protocol = protocolMatch[1];
  if (protocol === undefined) {
    // Return protocol completions.
    let completions: Completion[] = [];
    for (let [name, factory] of dataSourceFactories) {
      name = name + '://';
      if (name.startsWith(url)) {
        completions.push({value: name, description: factory.description});
      }
    }
    return Promise.resolve({offset: 0, completions});
  }
  let factory = dataSourceFactories.get(protocol);
  if (factory !== undefined) {
    let subCompleter = factory.volumeCompleter;
    if (subCompleter !== undefined) {
      return cancellableThen(
          subCompleter(protocolMatch[2]),
          completions => applyCompletionOffset(protocol.length + 3, completions));
    }
  }
  return Promise.reject<CompletionResult>(null);
}

export function suggestLayerName(url: string) {
  let [factories, path] = getDataSource(url);
  let suggestor = factories.suggestLayerName;
  if (suggestor !== undefined) {
    return suggestor(path);
  }
  return suggestLayerNameBasedOnSeparator(path);
}

export function findSourceGroup(url: string) {
  let [factories, path, dataSourceName] = getDataSource(url);
  let helper = factories.findSourceGroup || findSourceGroupBasedOnSeparator;
  return helper(path) + dataSourceName.length + 3;
}
