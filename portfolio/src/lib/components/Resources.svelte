<script lang="ts">
	import { dataStore } from '$lib/stores/dataStore';
	import DocumentCard from '$lib/components/DocumentCard.svelte';

	let documents = $dataStore.DOCUMENTS;
</script>

<h2 class="m-0 mb-12 font-mono text-2xl font-semibold uppercase tracking-widest text-black">
	/Resources
</h2>
<div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-y-8 gap-x-12">
	{#each documents as document, i}
			<DocumentCard
				title={document.title}
				src={document.src}
				download={document.download}
				type={document.type}
				description={document.description}
			/>
	{/each}
</div>
