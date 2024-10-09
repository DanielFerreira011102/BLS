<script>
    // Import the data from data.ts file from the static folder
    import DATA from '../data';
    import '../app.css';
    import Footer from "$lib/components/Footer.svelte";
    import Header from "$lib/components/Header.svelte";
</script>

<section id="layout" class="bg-white px-4 sm:px-6 py-2.5 container mx-auto">
    <Header personal={DATA.PERSONAL} navlist={DATA.NAVLIST} />
    <slot/>
    <Footer meta={DATA.META} personal={DATA.PERSONAL} contact={DATA.CONTACT} />
</section>