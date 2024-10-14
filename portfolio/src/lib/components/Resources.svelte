<script lang="ts">
	import { dataStore } from '$lib/stores/dataStore';
	import DocumentCard from '$lib/components/DocumentCard.svelte';

	let documents = $dataStore.DOCUMENTS;
</script>


<h2 class="m-0 mb-12 font-mono text-2xl font-semibold uppercase tracking-widest text-black">
	/Resources
</h2>
<ul>
	{#each documents as document, i}
		<li>
			<DocumentCard
				title={document.title}
				src={document.src}
				download={document.download}
				type={document.type}
				description={document.description}
			/>
		</li>
	{/each}
</ul>
