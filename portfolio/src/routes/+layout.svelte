<script>
    import '../app.css';
    import Footer from "$lib/components/Footer.svelte";
    import Header from "$lib/components/Header.svelte";
</script>

<section id="layout" class="container mx-auto bg-white px-4 sm:px-6 py-2.5">
    <Header />
    <slot/>
    <Footer />
</section>