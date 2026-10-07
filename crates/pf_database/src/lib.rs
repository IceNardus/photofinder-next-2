//! # pf_database — SQLite + migration + repository
//!
//! 不依赖 `pf_ai` / `pf_vector` / `pf_application`（架构约束）。
//!
//! 设计：
//! - `r2d2::Pool<SqliteConnectionManager>` 连接池
//! - WAL 模式 + `synchronous=NORMAL`
//! - Repository 模式，每个领域类型一个
//! - Migration 纯 SQL 文件

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod connection;
pub mod error;
pub mod migration;
pub mod repositories;

pub use connection::{Database, Transaction};
pub use error::DatabaseError;
pub use migration::{builtin_migrations, Migration, MigrationSet};
pub use repositories::{
    bodies::{BodyRow, BodiesRepository, NewBody},
    face::{FaceIndexStatus, FaceRepository, FaceRow, FaceStatus, NewFace},
    face_person_matches::{
        FacePersonAssignmentRepository, FacePersonAssignmentRow, MatchStatus,
        NewFacePersonAssignment,
    },
    image::{ImageRepository, ImageRow, NewImage, ScanStatus, ThumbnailStatus},
    object::{NewObject, ObjectRepository, ObjectRow, RoiType},
    person::{NewPerson, PersonIdentityStatus, PersonRepository, PersonRow, PersonStatus},
    person_body_prototypes::{NewPersonBodyPrototype, PersonBodyPrototypeRepository, PersonBodyPrototypeRow},
    person_prototypes::{NewPersonPrototype, PersonPrototypeRepository, PersonPrototypeRow, PrototypeType},
    shadow_records::{NewShadowRecord, ShadowRecordRow, ShadowRecordsRepository},
    task::{NewTask, TaskRepository, TaskRow, TaskStatus as DbTaskStatus},
};