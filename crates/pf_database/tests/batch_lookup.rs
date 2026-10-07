//! Audit B1/B3: `list_by_vector_ids` 批量反查的正确性。
//!
//! 覆盖:
//! - 命中部分 vector_id 时返回对应行
//! - 不存在的 vector_id 被跳过
//! - 超过 SQLite 单语句参数分块阈值(500)时仍完整返回

use pf_database::{builtin_migrations, Database, NewImage, NewObject, RoiType};

fn open_fresh_db() -> Database {
    Database::open_in_memory(builtin_migrations()).expect("open in-memory")
}

fn make_new_image(path: &str) -> NewImage {
    NewImage {
        path: path.into(),
        hash: "abc".into(),
        size: 1024,
        modified_time: 0,
        width: 800,
        height: 600,
        captured_at: Some(chrono::Utc::now()),
    }
}

fn make_new_object(image_id: i64, vector_id: i64) -> NewObject {
    NewObject {
        image_id,
        class_id: 1,
        class_name: "test".into(),
        confidence: 0.9,
        bbox: pf_core::BBox::new(0.0, 0.0, 100.0, 100.0),
        model_version: "v1".into(),
        vector_id,
        roi_scale: 0.5,
        roi_type: RoiType::SlidingWindow,
    }
}

#[test]
fn list_by_vector_ids_returns_matching_rows_only() {
    let db = open_fresh_db();
    db.transaction(|tx| {
        let img_id = tx.images().insert(&make_new_image("/tmp/batch.jpg")).expect("img");
        tx.objects().insert(&make_new_object(img_id, 100)).expect("o100");
        tx.objects().insert(&make_new_object(img_id, 200)).expect("o200");
        tx.objects().insert(&make_new_object(img_id, 300)).expect("o300");
        Ok(())
    })
    .expect("tx");

    let rows = db
        .transaction(|tx| tx.objects().list_by_vector_ids(&[200, 300, 999]))
        .expect("batch");
    let mut vids: Vec<i64> = rows.iter().map(|o| o.vector_id).collect();
    vids.sort();
    assert_eq!(vids, vec![200, 300], "999 missing → skipped");
}

#[test]
fn list_by_vector_ids_handles_more_than_500_ids() {
    let db = open_fresh_db();
    let n = 1200;
    db.transaction(|tx| {
        let img_id = tx.images().insert(&make_new_image("/tmp/chunk.jpg")).expect("img");
        for i in 0..n {
            tx.objects().insert(&make_new_object(img_id, i as i64)).expect("obj");
        }
        Ok(())
    })
    .expect("tx");

    let want: Vec<i64> = (0..n).collect();
    let rows = db
        .transaction(|tx| tx.objects().list_by_vector_ids(&want))
        .expect("batch");
    assert_eq!(rows.len(), n as usize, "all {} rows returned across chunk boundary", n);
    let mut got: Vec<i64> = rows.iter().map(|o| o.vector_id).collect();
    got.sort();
    assert_eq!(got, want);
}

#[test]
fn list_by_ids_handles_more_than_500_ids() {
    let db = open_fresh_db();
    let n = 1200;
    db.transaction(|tx| {
        let img_id = tx.images().insert(&make_new_image("/tmp/chunk2.jpg")).expect("img");
        for i in 0..n {
            tx.objects().insert(&make_new_object(img_id, i as i64)).expect("obj");
        }
        Ok(())
    })
    .expect("tx");

    let rows = db
        .transaction(|tx| tx.objects().list_by_ids(&(1..=n).collect::<Vec<i64>>()))
        .expect("batch");
    assert_eq!(rows.len(), n as usize, "all {} rows returned across chunk boundary", n);
    let mut got: Vec<i64> = rows.iter().map(|o| o.id).collect();
    got.sort();
    assert_eq!(got, (1..=n).collect::<Vec<i64>>());
}
