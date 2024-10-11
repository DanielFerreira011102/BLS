import { readable } from 'svelte/store';
import DATA from '../../data';

export const dataStore = readable(DATA);
