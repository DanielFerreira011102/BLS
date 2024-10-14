<script lang="ts">
    import { Icon } from 'svelte-icons-pack';
	import { RiSystemDownloadLine } from 'svelte-icons-pack/ri';
	import presentation from '$lib/images/slides.png';

	export let title: string;
	export let src: string;
	export let download = true;
	export let type: string;
	export let description: string;

	const images = {
		presentation: presentation
	};
</script>

<a
	target="_blank"
	rel="noopener noreferrer"
	href={download ? src : undefined}
	download={title}
	class="mb-5 mr-5 flex h-52 w-56 cursor-pointer flex-col overflow-hidden rounded-md border border-gray-200 bg-white shadow"
>
	<div class="flex flex-grow items-center justify-center">
		{#if type === 'presentation'}
			<img src={images.presentation} alt="presentation" class="h-28 w-28" />
		{/if}
	</div>
	<div class="flex items-center justify-between h-12 px-3">
		<span class="overflow-hidden text-ellipsis whitespace-nowrap text-sm font-semibold">
			{title}
		</span>
		{#if download}
			<Icon src={RiSystemDownloadLine} size="24" color="#9ca3af" />
		{/if}
	</div>
</a>
